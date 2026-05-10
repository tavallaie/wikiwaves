"""Tests for the WikiWaves Fetcher module."""

from __future__ import annotations

import datetime
import os
import unittest
import urllib.parse
from typing import Any
from unittest import mock

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
# Test Cases
# ---------------------------------------------------------------------------


class TestFetcher(unittest.TestCase):
    def setUp(self) -> None:
        """Set up dummy credentials for every test."""
        os.environ["WM_APP_NAME"] = "WikiWavesTest/1.0"
        os.environ["WM_CONTACT"] = "test@example.com"
        os.environ["WM_ACCESS_TOKEN"] = "fake-token"
        self.fetcher = WikiFetcher()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def test_init_missing_env(self) -> None:
        """Missing required env vars must raise RuntimeError immediately."""
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError) as ctx:
                WikiFetcher(env_path="/nonexistent")
        self.assertIn("Missing required environment", str(ctx.exception))

    def test_init_explicit_args_override_env(self) -> None:
        """Explicit constructor args should win over environment."""
        os.environ["WM_APP_NAME"] = "Wrong"
        os.environ["WM_CONTACT"] = "wrong@example.com"
        f = WikiFetcher(app_name="Right", contact="right@example.com")
        self.assertEqual(f.app_name, "Right")
        self.assertEqual(f.contact, "right@example.com")

    # ------------------------------------------------------------------
    # On This Day
    # ------------------------------------------------------------------

    @responses.activate
    def test_fetch_on_this_day_success(self) -> None:
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

        events = self.fetcher.fetch_on_this_day(datetime.date(2024, 1, 15))
        self.assertEqual(len(events), 2)

        evt0 = events[0]
        self.assertIsInstance(evt0, OnThisDayEvent)
        self.assertEqual(evt0.year, 1821)
        self.assertEqual(evt0.description, "Napoleon died in exile.")
        self.assertEqual(evt0.related_titles, ["Napoleon"])
        self.assertEqual(evt0.event_type, "selected")

        evt1 = events[1]
        self.assertEqual(evt1.year, 1929)
        self.assertEqual(evt1.event_type, "births")

    @responses.activate
    def test_fetch_on_this_day_malformed_item_skipped(self) -> None:
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

        events = self.fetcher.fetch_on_this_day(datetime.date(2024, 1, 15))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].year, 2000)

    # ------------------------------------------------------------------
    # Recent Changes
    # ------------------------------------------------------------------

    @responses.activate
    def test_fetch_recent_changes_success(self) -> None:
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

        edits = self.fetcher.fetch_recent_changes(hours=24)
        self.assertEqual(len(edits), 2)

        e0 = edits[0]
        self.assertIsInstance(e0, TrendingEdit)
        self.assertEqual(e0.title, "Python (programming language)")
        self.assertEqual(e0.bytes_changed, 200)
        self.assertFalse(e0.is_bot)
        self.assertFalse(e0.is_minor)

        e1 = edits[1]
        self.assertTrue(e1.is_bot)
        self.assertTrue(e1.is_minor)

    # ------------------------------------------------------------------
    # Page Content
    # ------------------------------------------------------------------

    def _query_response(self, pages: dict[str, Any]) -> dict[str, Any]:
        """Helper to build a standard action=query response."""
        return {
            "query": {
                "pages": {str(i): page for i, page in enumerate(pages.values())}
            }
        }

    @responses.activate
    def test_fetch_page_success(self) -> None:
        """Single-page fetch must return a fully populated WikiPage."""
        responses.add(
            responses.GET,
            "https://en.wikipedia.org/w/api.php",
            json=self._query_response(
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

        page = self.fetcher.fetch_page("Napoleon")
        self.assertIsNotNone(page)
        self.assertEqual(page.title, "Napoleon")
        self.assertIn("French general", page.plain_text)
        self.assertEqual(page.internal_links, ["France", "Saint Helena"])
        self.assertEqual(page.external_links, ["https://example.com/napoleon"])
        self.assertEqual(page.source_url, "https://en.wikipedia.org/wiki/Napoleon")
        self.assertGreater(page.word_count, 0)
        self.assertTrue(page.is_featured)
        self.assertFalse(page.is_good)
        self.assertIn("<p>", page.html)

    @responses.activate
    def test_fetch_pages_bulk(self) -> None:
        """Bulk fetch must handle multiple titles in one call."""
        responses.add(
            responses.GET,
            "https://en.wikipedia.org/w/api.php",
            json=self._query_response(
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

        results = self.fetcher.fetch_pages(["Napoleon", "Waterloo"])
        self.assertEqual(len(results), 2)
        self.assertIsNotNone(results["Napoleon"])
        self.assertIsNotNone(results["Waterloo"])
        self.assertTrue(results["Waterloo"].is_good)

    @responses.activate
    def test_fetch_page_missing_returns_none(self) -> None:
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

        page = self.fetcher.fetch_page("ThisPageDoesNotExist")
        self.assertIsNone(page)

    @responses.activate
    def test_fetch_page_redirect_followed(self) -> None:
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

        page = self.fetcher.fetch_page("USA")
        self.assertIsNotNone(page)
        self.assertEqual(page.title, "United States")

    # ------------------------------------------------------------------
    # Page Views
    # ------------------------------------------------------------------

    def _pageviews_url(self, title: str, days: int) -> str:
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
    def test_fetch_page_views_success(self) -> None:
        """Average daily views must be calculated correctly."""
        url = self._pageviews_url("Napoleon", 30)
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

        avg = self.fetcher.fetch_page_views("Napoleon", days=30)
        self.assertEqual(avg, 2000)

    @responses.activate
    def test_fetch_page_views_empty_returns_none(self) -> None:
        """Empty page-views response must return ``None``."""
        url = self._pageviews_url("Napoleon", 30)
        responses.add(
            responses.GET,
            url,
            json={"items": []},
            status=200,
        )

        avg = self.fetcher.fetch_page_views("Napoleon", days=30)
        self.assertIsNone(avg)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    @responses.activate
    def test_search_returns_title(self) -> None:
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

        result = self.fetcher.search("python programming")
        self.assertEqual(result, "Python (programming language)")

    @responses.activate
    def test_search_no_results_returns_none(self) -> None:
        """Empty search results must return ``None``."""
        responses.add(
            responses.GET,
            "https://en.wikipedia.org/w/api.php",
            json={"query": {"search": []}},
            status=200,
        )

        self.assertIsNone(self.fetcher.search("xyzabc123nonsense"))

    # ------------------------------------------------------------------
    # Article Quality
    # ------------------------------------------------------------------

    @responses.activate
    def test_check_article_quality(self) -> None:
        """Quality check must delegate to fetch_pages and return booleans."""
        responses.add(
            responses.GET,
            "https://en.wikipedia.org/w/api.php",
            json=self._query_response(
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

        featured, good = self.fetcher.check_article_quality("Napoleon")
        self.assertTrue(featured)
        self.assertFalse(good)

    # ------------------------------------------------------------------
    # Retry / Error handling
    # ------------------------------------------------------------------

    @responses.activate
    def test_retry_on_timeout_then_success(self) -> None:
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
        self.fetcher._backoff = 0.01
        events = self.fetcher.fetch_on_this_day(datetime.date(2024, 1, 15))
        self.assertEqual(events, [])
        self.assertEqual(len(responses.calls), 2)

    @responses.activate
    def test_rate_limit_with_retry_after(self) -> None:
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

        events = self.fetcher.fetch_on_this_day(datetime.date(2024, 1, 15))
        self.assertEqual(events, [])
        self.assertEqual(len(responses.calls), 2)

    @responses.activate
    def test_rate_limit_exhausted_raises(self) -> None:
        """Repeated 429s beyond the retry limit must raise RateLimitError."""
        for _ in range(self.fetcher._retries + 1):
            responses.add(
                responses.GET,
                "https://api.wikimedia.org/feed/v1/wikipedia/en/featured/2024/01/15",
                status=429,
            )

        self.fetcher._backoff = 0.01
        with self.assertRaises(RateLimitError):
            self.fetcher.fetch_on_this_day(datetime.date(2024, 1, 15))

    @responses.activate
    def test_page_not_found_raises(self) -> None:
        """A 404 must raise PageNotFoundError immediately (no retry)."""
        responses.add(
            responses.GET,
            "https://api.wikimedia.org/feed/v1/wikipedia/en/featured/2024/01/15",
            body="Not Found",
            status=404,
            content_type="text/plain",
        )

        with self.assertRaises(PageNotFoundError):
            self.fetcher.fetch_on_this_day(datetime.date(2024, 1, 15))

    @responses.activate
    def test_malformed_json_raises(self) -> None:
        """Non-JSON body must raise MalformedResponseError."""
        responses.add(
            responses.GET,
            "https://api.wikimedia.org/feed/v1/wikipedia/en/featured/2024/01/15",
            body="not json",
            status=200,
        )

        with self.assertRaises(MalformedResponseError):
            self.fetcher.fetch_on_this_day(datetime.date(2024, 1, 15))

    # ------------------------------------------------------------------
    # Page Revisions
    # ------------------------------------------------------------------

    @responses.activate
    def test_fetch_page_revisions_counts_in_range(self) -> None:
        """Only revisions within the cutoff window must be counted."""
        now = datetime.datetime.now(datetime.timezone.utc)
        recent = now.isoformat().replace("+00:00", "Z")
        old = (now - datetime.timedelta(days=60)).isoformat().replace("+00:00", "Z")

        responses.add(
            responses.GET,
            "https://en.wikipedia.org/w/api.php",
            json=self._query_response(
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

        count = self.fetcher.fetch_page_revisions("Napoleon", days=30)
        self.assertEqual(count, 2)

    @responses.activate
    def test_fetch_page_revisions_missing_page(self) -> None:
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

        self.assertIsNone(self.fetcher.fetch_page_revisions("Missing", days=30))

    # ------------------------------------------------------------------
    # Enrich Pages
    # ------------------------------------------------------------------

    @responses.activate
    def test_enrich_pages_success(self) -> None:
        """enrich_pages must aggregate content, views, and revision counts."""
        # 1) Batch page content
        responses.add(
            responses.GET,
            "https://en.wikipedia.org/w/api.php",
            json=self._query_response(
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
        views_url = self._pageviews_url("Napoleon", 30)
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
            json=self._query_response(
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

        results = self.fetcher.enrich_pages(["Napoleon"], days=30)
        self.assertEqual(len(results), 1)

        metrics = results["Napoleon"]
        self.assertIsNotNone(metrics)
        self.assertEqual(metrics.title, "Napoleon")
        self.assertEqual(metrics.word_count, 5)
        self.assertEqual(metrics.internal_links, 2)
        self.assertEqual(metrics.external_links, 1)
        self.assertEqual(metrics.page_views_30d, 1500)
        self.assertEqual(metrics.edit_count_30d, 2)
        self.assertTrue(metrics.is_featured)
        self.assertFalse(metrics.is_good)
        self.assertEqual(metrics.source_url, "https://en.wikipedia.org/wiki/Napoleon")

    @responses.activate
    def test_enrich_pages_missing_page(self) -> None:
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

        results = self.fetcher.enrich_pages(["MissingPage"], days=30)
        self.assertIsNone(results["MissingPage"])


if __name__ == "__main__":
    unittest.main()
