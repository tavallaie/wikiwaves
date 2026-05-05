"""Tests for the WikiWaves Fetcher module."""

from __future__ import annotations

import datetime
import urllib.parse
from typing import Any

import pytest
import requests
import responses

from wikiwaves.fetcher import (
    FetcherError,
    MalformedResponseError,
    NetworkError,
    OnThisDayEvent,
    PageNotFoundError,
    RateLimitError,
    TrendingEdit,
    WikiFetcher,
    WikiPage,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fetcher(monkeypatch: pytest.MonkeyPatch) -> WikiFetcher:
    """Return a WikiFetcher instance with dummy credentials."""
    monkeypatch.setenv("WM_APP_NAME", "WikiWavesTest/1.0")
    monkeypatch.setenv("WM_CONTACT", "test@example.com")
    monkeypatch.setenv("WM_ACCESS_TOKEN", "fake-token")
    return WikiFetcher()


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


def test_init_missing_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing required env vars must raise RuntimeError immediately."""
    monkeypatch.delenv("WM_APP_NAME", raising=False)
    monkeypatch.delenv("WM_CONTACT", raising=False)
    monkeypatch.delenv("WM_ACCESS_TOKEN", raising=False)
    # Prevent auto-loading of .env file
    with pytest.raises(RuntimeError, match="Missing required environment"):
        WikiFetcher(env_path="/nonexistent")


def test_init_explicit_args_override_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit constructor args should win over environment."""
    monkeypatch.setenv("WM_APP_NAME", "Wrong")
    monkeypatch.setenv("WM_CONTACT", "wrong@example.com")
    f = WikiFetcher(app_name="Right", contact="right@example.com")
    assert f.app_name == "Right"
    assert f.contact == "right@example.com"


# ---------------------------------------------------------------------------
# On This Day
# ---------------------------------------------------------------------------


@responses.activate
def test_fetch_on_this_day_success(fetcher: WikiFetcher) -> None:
    """Happy path: parse the REST feed into OnThisDayEvent objects."""
    responses.add(
        responses.GET,
        "https://api.wikimedia.org/feed/v1/wikipedia/en/featured/2024/01/15",
        json={
            "onthisday": {
                "selected": [
                    {
                        "year": 1821,
                        "text": "Napoleon died in exile.",
                        "pages": [
                            {
                                "titles": {"canonical": "Napoleon"},
                                "extract": "Napoleon Bonaparte was a French general.",
                            }
                        ],
                    }
                ],
                "births": [
                    {
                        "year": 1929,
                        "text": "Martin Luther King Jr. was born.",
                        "pages": [],
                    }
                ],
            }
        },
        status=200,
    )

    events = fetcher.fetch_on_this_day(datetime.date(2024, 1, 15))
    assert len(events) == 2

    evt0 = events[0]
    assert isinstance(evt0, OnThisDayEvent)
    assert evt0.year == 1821
    assert evt0.description == "Napoleon died in exile."
    assert evt0.related_titles == ["Napoleon"]
    assert evt0.event_type == "selected"

    evt1 = events[1]
    assert evt1.year == 1929
    assert evt1.event_type == "births"


@responses.activate
def test_fetch_on_this_day_malformed_item_skipped(fetcher: WikiFetcher) -> None:
    """A malformed item inside the feed must be skipped, not crash the pipeline."""
    responses.add(
        responses.GET,
        "https://api.wikimedia.org/feed/v1/wikipedia/en/featured/2024/01/15",
        json={
            "onthisday": {
                "events": [
                    {"year": "not-an-int", "text": "Bad item"},
                    {"year": 2000, "text": "Good item", "pages": []},
                ]
            }
        },
        status=200,
    )

    events = fetcher.fetch_on_this_day(datetime.date(2024, 1, 15))
    assert len(events) == 1
    assert events[0].year == 2000


# ---------------------------------------------------------------------------
# Recent Changes
# ---------------------------------------------------------------------------


@responses.activate
def test_fetch_recent_changes_success(fetcher: WikiFetcher) -> None:
    """Recent changes must be parsed into TrendingEdit objects."""
    responses.add(
        responses.GET,
        "https://en.wikipedia.org/w/api.php",
        json={
            "query": {
                "recentchanges": [
                    {
                        "title": "Python (programming language)",
                        "user": "CoderOne",
                        "timestamp": "2024-01-15T10:00:00Z",
                        "oldlen": 1000,
                        "newlen": 1200,
                        "comment": "Added syntax section",
                        "flags": [],
                        "tags": [],
                    },
                    {
                        "title": "Bot test",
                        "user": "BotAccount",
                        "timestamp": "2024-01-15T09:00:00Z",
                        "oldlen": 500,
                        "newlen": 550,
                        "comment": "Cleanup",
                        "flags": ["bot", "minor"],
                        "tags": [],
                    },
                ]
            }
        },
        status=200,
    )

    edits = fetcher.fetch_recent_changes(hours=24)
    assert len(edits) == 2

    e0 = edits[0]
    assert isinstance(e0, TrendingEdit)
    assert e0.title == "Python (programming language)"
    assert e0.bytes_changed == 200
    assert e0.is_bot is False
    assert e0.is_minor is False

    e1 = edits[1]
    assert e1.is_bot is True
    assert e1.is_minor is True


# ---------------------------------------------------------------------------
# Page Content
# ---------------------------------------------------------------------------


def _query_response(pages: dict[str, Any]) -> dict[str, Any]:
    """Helper to build a standard action=query response."""
    return {
        "query": {
            "pages": {str(i): page for i, page in enumerate(pages.values())}
        }
    }


@responses.activate
def test_fetch_page_success(fetcher: WikiFetcher) -> None:
    """Single-page fetch must return a fully populated WikiPage."""
    responses.add(
        responses.GET,
        "https://en.wikipedia.org/w/api.php",
        json=_query_response(
            {
                "Napoleon": {
                    "title": "Napoleon",
                    "extract": "Napoleon Bonaparte was a French general and emperor.",
                    "links": [
                        {"ns": 0, "title": "France"},
                        {"ns": 0, "title": "Saint Helena"},
                        {"ns": 1, "title": "Talk:Napoleon"},
                    ],
                    "extlinks": ["https://example.com/napoleon"],
                    "fullurl": "https://en.wikipedia.org/wiki/Napoleon",
                    "categories": [
                        {"title": "Category:Featured articles"},
                    ],
                }
            }
        ),
        status=200,
    )
    # HTML fetch via action=parse
    responses.add(
        responses.GET,
        "https://en.wikipedia.org/w/api.php",
        json={
            "parse": {
                "text": {"*": "<p>Napoleon Bonaparte was a French general.</p>"}
            }
        },
        status=200,
    )

    page = fetcher.fetch_page("Napoleon")
    assert page is not None
    assert page.title == "Napoleon"
    assert "French general" in page.plain_text
    assert page.internal_links == ["France", "Saint Helena"]
    assert page.external_links == ["https://example.com/napoleon"]
    assert page.source_url == "https://en.wikipedia.org/wiki/Napoleon"
    assert page.word_count > 0
    assert page.is_featured is True
    assert page.is_good is False
    assert "<p>" in page.html


@responses.activate
def test_fetch_pages_bulk(fetcher: WikiFetcher) -> None:
    """Bulk fetch must handle multiple titles in one call."""
    responses.add(
        responses.GET,
        "https://en.wikipedia.org/w/api.php",
        json=_query_response(
            {
                "Napoleon": {
                    "title": "Napoleon",
                    "extract": "French general.",
                    "links": [],
                    "extlinks": [],
                    "fullurl": "https://en.wikipedia.org/wiki/Napoleon",
                    "categories": [],
                },
                "Waterloo": {
                    "title": "Waterloo",
                    "extract": "A battle in 1815.",
                    "links": [],
                    "extlinks": [],
                    "fullurl": "https://en.wikipedia.org/wiki/Waterloo",
                    "categories": [{"title": "Category:Good articles"}],
                },
            }
        ),
        status=200,
    )

    results = fetcher.fetch_pages(["Napoleon", "Waterloo"])
    assert len(results) == 2
    assert results["Napoleon"] is not None
    assert results["Waterloo"] is not None
    assert results["Waterloo"].is_good is True


@responses.activate
def test_fetch_page_missing_returns_none(fetcher: WikiFetcher) -> None:
    """A missing page must return ``None``, not raise."""
    responses.add(
        responses.GET,
        "https://en.wikipedia.org/w/api.php",
        json={
            "query": {
                "pages": {
                    "-1": {
                        "title": "ThisPageDoesNotExist",
                        "missing": "",
                    }
                }
            }
        },
        status=200,
    )

    page = fetcher.fetch_page("ThisPageDoesNotExist")
    assert page is None


@responses.activate
def test_fetch_page_redirect_followed(fetcher: WikiFetcher) -> None:
    """Redirects must be resolved automatically."""
    responses.add(
        responses.GET,
        "https://en.wikipedia.org/w/api.php",
        json={
            "query": {
                "normalized": [{"from": "USA", "to": "USA"}],
                "redirects": [{"from": "USA", "to": "United States"}],
                "pages": {
                    "1": {
                        "title": "United States",
                        "extract": "The United States is a country.",
                        "links": [],
                        "extlinks": [],
                        "fullurl": "https://en.wikipedia.org/wiki/United_States",
                        "categories": [],
                    }
                },
            }
        },
        status=200,
    )
    responses.add(
        responses.GET,
        "https://en.wikipedia.org/w/api.php",
        json={
            "parse": {
                "text": {"*": "<p>The United States is a country.</p>"}
            }
        },
        status=200,
    )

    page = fetcher.fetch_page("USA")
    assert page is not None
    assert page.title == "United States"


# ---------------------------------------------------------------------------
# Page Views
# ---------------------------------------------------------------------------


def _pageviews_url(title: str, days: int) -> str:
    """Build the expected PageViews API URL for *today*."""
    end = datetime.date.today() - datetime.timedelta(days=1)
    start = end - datetime.timedelta(days=days - 1)
    encoded = urllib.parse.quote(title.replace(" ", "_"), safe="")
    return (
        f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
        f"en.wikipedia/all-access/all-agents/{encoded}/"
        f"daily/{start.strftime('%Y%m%d')}/{end.strftime('%Y%m%d')}"
    )


@responses.activate
def test_fetch_page_views_success(fetcher: WikiFetcher) -> None:
    """Average daily views must be calculated correctly."""
    url = _pageviews_url("Napoleon", 30)
    responses.add(
        responses.GET,
        url,
        json={
            "items": [
                {"views": 1000},
                {"views": 2000},
                {"views": 3000},
            ]
        },
        status=200,
    )

    avg = fetcher.fetch_page_views("Napoleon", days=30)
    assert avg == 2000


@responses.activate
def test_fetch_page_views_empty_returns_none(fetcher: WikiFetcher) -> None:
    """Empty page-views response must return ``None``."""
    url = _pageviews_url("Napoleon", 30)
    responses.add(
        responses.GET,
        url,
        json={"items": []},
        status=200,
    )

    avg = fetcher.fetch_page_views("Napoleon", days=30)
    assert avg is None


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


@responses.activate
def test_search_returns_title(fetcher: WikiFetcher) -> None:
    """Search must return the canonical title of the top result."""
    responses.add(
        responses.GET,
        "https://en.wikipedia.org/w/api.php",
        json={
            "query": {
                "search": [
                    {"title": "Python (programming language)"},
                ]
            }
        },
        status=200,
    )

    result = fetcher.search("python programming")
    assert result == "Python (programming language)"


@responses.activate
def test_search_no_results_returns_none(fetcher: WikiFetcher) -> None:
    """Empty search results must return ``None``."""
    responses.add(
        responses.GET,
        "https://en.wikipedia.org/w/api.php",
        json={"query": {"search": []}},
        status=200,
    )

    assert fetcher.search("xyzabc123nonsense") is None


# ---------------------------------------------------------------------------
# Article Quality
# ---------------------------------------------------------------------------


@responses.activate
def test_check_article_quality(fetcher: WikiFetcher) -> None:
    """Quality check must delegate to fetch_pages and return booleans."""
    responses.add(
        responses.GET,
        "https://en.wikipedia.org/w/api.php",
        json=_query_response(
            {
                "Napoleon": {
                    "title": "Napoleon",
                    "extract": "...",
                    "links": [],
                    "extlinks": [],
                    "fullurl": "",
                    "categories": [{"title": "Category:Featured articles"}],
                }
            }
        ),
        status=200,
    )

    featured, good = fetcher.check_article_quality("Napoleon")
    assert featured is True
    assert good is False


# ---------------------------------------------------------------------------
# Retry / Error handling
# ---------------------------------------------------------------------------


@responses.activate
def test_retry_on_timeout_then_success(fetcher: WikiFetcher) -> None:
    """A single timeout followed by success must be retried transparently."""
    responses.add(
        responses.GET,
        "https://api.wikimedia.org/feed/v1/wikipedia/en/featured/2024/01/15",
        body=requests.Timeout("Connection timed out"),
    )
    responses.add(
        responses.GET,
        "https://api.wikimedia.org/feed/v1/wikipedia/en/featured/2024/01/15",
        json={"onthisday": {"selected": []}},
        status=200,
    )

    # Reduce backoff so the test runs quickly
    fetcher._backoff = 0.01
    events = fetcher.fetch_on_this_day(datetime.date(2024, 1, 15))
    assert events == []
    assert len(responses.calls) == 2


@responses.activate
def test_rate_limit_with_retry_after(fetcher: WikiFetcher) -> None:
    """A 429 with Retry-After must be respected and then succeed."""
    responses.add(
        responses.GET,
        "https://api.wikimedia.org/feed/v1/wikipedia/en/featured/2024/01/15",
        status=429,
        headers={"Retry-After": "1"},
    )
    responses.add(
        responses.GET,
        "https://api.wikimedia.org/feed/v1/wikipedia/en/featured/2024/01/15",
        json={"onthisday": {"selected": []}},
        status=200,
    )

    events = fetcher.fetch_on_this_day(datetime.date(2024, 1, 15))
    assert events == []
    assert len(responses.calls) == 2


@responses.activate
def test_rate_limit_exhausted_raises(fetcher: WikiFetcher) -> None:
    """Repeated 429s beyond the retry limit must raise RateLimitError."""
    for _ in range(fetcher._retries + 1):
        responses.add(
            responses.GET,
            "https://api.wikimedia.org/feed/v1/wikipedia/en/featured/2024/01/15",
            status=429,
        )

    fetcher._backoff = 0.01
    with pytest.raises(RateLimitError):
        fetcher.fetch_on_this_day(datetime.date(2024, 1, 15))


@responses.activate
def test_page_not_found_raises(fetcher: WikiFetcher) -> None:
    """A 404 must raise PageNotFoundError immediately (no retry)."""
    responses.add(
        responses.GET,
        "https://api.wikimedia.org/feed/v1/wikipedia/en/featured/2024/01/15",
        body="Not Found",
        status=404,
        content_type="text/plain",
    )

    with pytest.raises(PageNotFoundError):
        fetcher.fetch_on_this_day(datetime.date(2024, 1, 15))


@responses.activate
def test_malformed_json_raises(fetcher: WikiFetcher) -> None:
    """Non-JSON body must raise MalformedResponseError."""
    responses.add(
        responses.GET,
        "https://api.wikimedia.org/feed/v1/wikipedia/en/featured/2024/01/15",
        body="not json",
        status=200,
    )

    with pytest.raises(MalformedResponseError):
        fetcher.fetch_on_this_day(datetime.date(2024, 1, 15))


# ---------------------------------------------------------------------------
# Page Revisions
# ---------------------------------------------------------------------------


@responses.activate
def test_fetch_page_revisions_counts_in_range(fetcher: WikiFetcher) -> None:
    """Only revisions within the cutoff window must be counted."""
    now = datetime.datetime.now(datetime.timezone.utc)
    recent = now.isoformat().replace("+00:00", "Z")
    old = (now - datetime.timedelta(days=60)).isoformat().replace("+00:00", "Z")

    responses.add(
        responses.GET,
        "https://en.wikipedia.org/w/api.php",
        json=_query_response(
            {
                "Napoleon": {
                    "title": "Napoleon",
                    "revisions": [
                        {"timestamp": recent},
                        {"timestamp": recent},
                        {"timestamp": old},
                    ],
                }
            }
        ),
        status=200,
    )

    count = fetcher.fetch_page_revisions("Napoleon", days=30)
    assert count == 2


@responses.activate
def test_fetch_page_revisions_missing_page(fetcher: WikiFetcher) -> None:
    """A missing page must return ``None``."""
    responses.add(
        responses.GET,
        "https://en.wikipedia.org/w/api.php",
        json={
            "query": {
                "pages": {
                    "-1": {
                        "title": "Missing",
                        "missing": "",
                    }
                }
            }
        },
        status=200,
    )

    assert fetcher.fetch_page_revisions("Missing", days=30) is None


# ---------------------------------------------------------------------------
# Enrich Pages
# ---------------------------------------------------------------------------


@responses.activate
def test_enrich_pages_success(fetcher: WikiFetcher) -> None:
    """enrich_pages must aggregate content, views, and revision counts."""
    # 1) Batch page content
    responses.add(
        responses.GET,
        "https://en.wikipedia.org/w/api.php",
        json=_query_response(
            {
                "Napoleon": {
                    "title": "Napoleon",
                    "extract": "Napoleon was a French general.",
                    "links": [
                        {"ns": 0, "title": "France"},
                        {"ns": 0, "title": "Saint Helena"},
                    ],
                    "extlinks": ["https://example.com"],
                    "fullurl": "https://en.wikipedia.org/wiki/Napoleon",
                    "categories": [{"title": "Category:Featured articles"}],
                }
            }
        ),
        status=200,
    )

    # 2) Page views
    views_url = _pageviews_url("Napoleon", 30)
    responses.add(
        responses.GET,
        views_url,
        json={"items": [{"views": 1000}, {"views": 2000}]},
        status=200,
    )

    # 3) Revisions
    now = datetime.datetime.now(datetime.timezone.utc)
    recent = now.isoformat().replace("+00:00", "Z")
    responses.add(
        responses.GET,
        "https://en.wikipedia.org/w/api.php",
        json=_query_response(
            {
                "Napoleon": {
                    "title": "Napoleon",
                    "revisions": [
                        {"timestamp": recent},
                        {"timestamp": recent},
                    ],
                }
            }
        ),
        status=200,
    )

    results = fetcher.enrich_pages(["Napoleon"], days=30)
    assert len(results) == 1

    metrics = results["Napoleon"]
    assert metrics is not None
    assert metrics.title == "Napoleon"
    assert metrics.word_count == 5
    assert metrics.internal_links == 2
    assert metrics.external_links == 1
    assert metrics.page_views_30d == 1500
    assert metrics.edit_count_30d == 2
    assert metrics.is_featured is True
    assert metrics.is_good is False
    assert metrics.source_url == "https://en.wikipedia.org/wiki/Napoleon"


@responses.activate
def test_enrich_pages_missing_page(fetcher: WikiFetcher) -> None:
    """A missing page must map to ``None``."""
    responses.add(
        responses.GET,
        "https://en.wikipedia.org/w/api.php",
        json={
            "query": {
                "pages": {
                    "-1": {
                        "title": "MissingPage",
                        "missing": "",
                    }
                }
            }
        },
        status=200,
    )

    results = fetcher.enrich_pages(["MissingPage"], days=30)
    assert results["MissingPage"] is None
