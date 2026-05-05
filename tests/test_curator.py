"""Tests for the WikiWaves Curator module."""

from __future__ import annotations

import datetime

import pytest

from wikiwaves.curator import (
    AggregatedEvent,
    CuratedEpisode,
    TopicScore,
    aggregate_events,
    curate,
    score_event,
    select_topics,
)
from wikiwaves.fetcher.models import OnThisDayEvent, PageMetrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _page(title: str, word_count: int = 0, views: int | None = None, edits: int | None = None,
          internal: int = 0, external: int = 0, featured: bool = False, good: bool = False) -> PageMetrics:
    return PageMetrics(
        title=title,
        word_count=word_count,
        internal_links=internal,
        external_links=external,
        page_views_30d=views,
        edit_count_30d=edits,
        is_featured=featured,
        is_good=good,
    )


def _event(event_id: str, year: int | None, desc: str, pages: list[PageMetrics], etype: str = "selected") -> AggregatedEvent:
    return AggregatedEvent(event_id=event_id, year=year, description=desc, event_type=etype, related_pages=pages)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def test_aggregate_events_skips_missing_pages() -> None:
    events = [
        OnThisDayEvent(event_id="test-123", year=2000, description="Test", related_titles=["A", "B"], event_type="selected"),
    ]
    enriched = {"A": _page("A", word_count=100)}
    result = aggregate_events(events, enriched)
    assert len(result) == 1
    assert len(result[0].related_pages) == 1
    assert result[0].related_pages[0].title == "A"


def test_aggregate_events_empty_input() -> None:
    assert aggregate_events([], {}) == []


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def test_score_event_richness() -> None:
    evt = _event("evt-1", 2000, "Test", [_page("X", word_count=10_000, internal=500)])
    score = score_event(evt)
    assert score.content_richness == pytest.approx(1.0, 0.01)


def test_score_event_recency_modern() -> None:
    evt = _event("evt-2", 2020, "Test", [])
    assert score_event(evt).recency == 1.0


def test_score_event_recency_ancient() -> None:
    evt = _event("evt-3", 100, "Test", [])
    assert score_event(evt).recency == 0.2


def test_score_event_popularity() -> None:
    evt = _event("evt-4", 2000, "Test", [
        _page("A", views=10_000),
        _page("B", views=0),
    ])
    score = score_event(evt)
    assert score.popularity == pytest.approx(1.0, 0.01)


def test_score_event_quality() -> None:
    evt = _event("evt-5", 2000, "Test", [
        _page("A", featured=True),
        _page("B", good=True),
    ])
    score = score_event(evt)
    assert score.quality == pytest.approx(0.75, 0.01)


# ---------------------------------------------------------------------------
# Topic Selection
# ---------------------------------------------------------------------------


def test_select_topics_picks_top_by_score() -> None:
    events = [
        _event("evt-6", 2020, "Modern popular", [_page("A", word_count=10_000, views=50_000)]),
        _event("evt-7", 1800, "Old sparse", [_page("B", word_count=100, views=10)]),
    ]
    result = select_topics(events, count=1)
    assert len(result) == 1
    assert result[0].year == 2020


def test_select_topics_enforces_diversity() -> None:
    events = [
        _event("evt-8", 2020, "A", [_page("A", word_count=10_000)]),
        _event("evt-9", 2021, "B", [_page("B", word_count=9_000)]),
        _event("evt-10", 2022, "C", [_page("C", word_count=8_000)]),
        _event("evt-11", 2023, "D", [_page("D", word_count=7_000)]),
        _event("evt-12", 1500, "E", [_page("E", word_count=6_000)]),
        _event("evt-13", 1600, "F", [_page("F", word_count=5_000)]),
    ]
    result = select_topics(events, count=5)
    assert len(result) == 5
    centuries = {e.year // 100 for e in result if e.year}  # type: ignore
    assert len(centuries) >= 2  # diversity enforced


def test_select_topics_backfill_when_too_strict() -> None:
    events = [
        _event("evt-14", 2020, "A", [_page("A", word_count=10_000)]),
        _event("evt-15", 2021, "B", [_page("B", word_count=9_000)]),
    ]
    result = select_topics(events, count=5)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# Full Pipeline
# ---------------------------------------------------------------------------


def test_curate_produces_episode() -> None:
    events = [
        _event("evt-16", 2020, "Event A", [_page("A", word_count=5_000, views=10_000)]),
        _event("evt-17", 1990, "Event B", [_page("B", word_count=3_000, views=5_000)]),
    ]

    episode = curate(events, "2024-01-01", topic_count=1)
    assert isinstance(episode, CuratedEpisode)
    assert len(episode.topics) == 1
    assert len(episode.scoring_breakdown) == 1
    assert episode.total_events_considered == 2
