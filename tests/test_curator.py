"""Tests for the WikiWaves Curator module."""

from __future__ import annotations

import unittest

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


def _page(
    title: str,
    word_count: int = 0,
    views: int | None = None,
    edits: int | None = None,
    internal: int = 0,
    external: int = 0,
    featured: bool = False,
    good: bool = False,
) -> PageMetrics:
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


def _event(
    event_id: str,
    year: int | None,
    desc: str,
    pages: list[PageMetrics],
    etype: str = "selected",
) -> AggregatedEvent:
    return AggregatedEvent(
        event_id=event_id, year=year, description=desc, event_type=etype, related_pages=pages
    )


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------


class TestAggregation(unittest.TestCase):
    def test_aggregate_events_skips_missing_pages(self) -> None:
        events = [
            OnThisDayEvent(
                event_id="test-123",
                year=2000,
                description="Test",
                related_titles=["A", "B"],
                event_type="selected",
            ),
        ]
        enriched = {"A": _page("A", word_count=100)}
        result = aggregate_events(events, enriched)
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0].related_pages), 1)
        self.assertEqual(result[0].related_pages[0].title, "A")

    def test_aggregate_events_empty_input(self) -> None:
        self.assertEqual(aggregate_events([], []), [])


class TestScoring(unittest.TestCase):
    def test_score_event_richness(self) -> None:
        evt = _event("evt-1", 2000, "Test", [_page("X", word_count=10_000, internal=500)])
        score = score_event(evt)
        self.assertAlmostEqual(score.content_richness, 1.0, places=2)

    def test_score_event_recency_modern(self) -> None:
        evt = _event("evt-2", 2020, "Test", [])
        self.assertEqual(score_event(evt).recency, 1.0)

    def test_score_event_recency_ancient(self) -> None:
        evt = _event("evt-3", 100, "Test", [])
        self.assertEqual(score_event(evt).recency, 0.2)

    def test_score_event_popularity(self) -> None:
        evt = _event("evt-4", 2000, "Test", [_page("A", views=10_000), _page("B", views=0)])
        score = score_event(evt)
        self.assertAlmostEqual(score.popularity, 1.0, places=2)

    def test_score_event_quality(self) -> None:
        evt = _event("evt-5", 2000, "Test", [_page("A", featured=True), _page("B", good=True)])
        score = score_event(evt)
        self.assertAlmostEqual(score.quality, 0.75, places=2)


class TestTopicSelection(unittest.TestCase):
    def test_select_topics_picks_top_by_score(self) -> None:
        events = [
            _event("evt-6", 2020, "Modern popular", [_page("A", word_count=10_000, views=50_000)]),
            _event("evt-7", 1800, "Old sparse", [_page("B", word_count=100, views=10)]),
        ]
        result = select_topics(events, count=1)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].year, 2020)

    def test_select_topics_enforces_diversity(self) -> None:
        events = [
            _event("evt-8", 2020, "A", [_page("A", word_count=10_000)]),
            _event("evt-9", 2021, "B", [_page("B", word_count=9_000)]),
            _event("evt-10", 2022, "C", [_page("C", word_count=8_000)]),
            _event("evt-11", 2023, "D", [_page("D", word_count=7_000)]),
            _event("evt-12", 1500, "E", [_page("E", word_count=6_000)]),
            _event("evt-13", 1600, "F", [_page("F", word_count=5_000)]),
        ]
        result = select_topics(events, count=5)
        self.assertEqual(len(result), 5)
        centuries = {e.year // 100 for e in result if e.year}  # type: ignore
        self.assertGreaterEqual(len(centuries), 2)  # diversity enforced

    def test_select_topics_backfill_when_too_strict(self) -> None:
        events = [
            _event("evt-14", 2020, "A", [_page("A", word_count=10_000)]),
            _event("evt-15", 2021, "B", [_page("B", word_count=9_000)]),
        ]
        result = select_topics(events, count=5)
        self.assertEqual(len(result), 2)


class TestFullPipeline(unittest.TestCase):
    def test_curate_produces_episode(self) -> None:
        events = [
            _event("evt-16", 2020, "Event A", [_page("A", word_count=5_000, views=10_000)]),
            _event("evt-17", 1990, "Event B", [_page("B", word_count=3_000, views=5_000)]),
        ]

        episode = curate(events, "2024-01-01", topic_count=1)
        self.assertIsInstance(episode, CuratedEpisode)
        self.assertEqual(len(episode.topics), 1)
        self.assertEqual(len(episode.scoring_breakdown), 1)
        self.assertEqual(episode.total_events_considered, 2)


if __name__ == "__main__":
    unittest.main()
