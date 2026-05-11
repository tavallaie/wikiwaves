"""Tests for the WikiWaves Enricher module."""

from __future__ import annotations

import json
import unittest

from wikiwaves.curator.models import AggregatedEvent
from wikiwaves.enricher import (
    EnrichedTopic,
    enrich_topic,
    enrich_topics,
    validate_suggestions,
)
from wikiwaves.llm import LLMClient, LLMError
from wikiwaves.enricher.expander import _parse_llm_json
from wikiwaves.enricher.prompts import build_content_prompt
from wikiwaves.fetcher.models import WikiPage, PageMetrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _page(title: str, word_count: int = 500) -> WikiPage:
    return WikiPage(
        title=title,
        plain_text=f"Extract for {title}",
        word_count=word_count,
    )


def _metric(title: str, word_count: int = 500) -> PageMetrics:
    return PageMetrics(title=title, word_count=word_count)


def _event(
    description: str,
    related: list[PageMetrics],
    year: int = 2000,
) -> AggregatedEvent:
    return AggregatedEvent(
        event_id="evt-test",
        year=year,
        description=description,
        event_type="selected",
        related_pages=related,
    )


class FakeLLMClient:
    """Stub LLM client that returns a predetermined response."""

    def __init__(
        self,
        response: dict | None = None,
        raise_error: Exception | None = None,
    ) -> None:
        self.response = response
        self.raise_error = raise_error

    def chat(self, messages, temperature=0.3):
        if self.raise_error:
            raise self.raise_error
        return json.dumps(self.response)


class FakeFetcher:
    """Stub fetcher that returns pages from a pre-seeded dictionary."""

    def __init__(self, pages: dict[str, WikiPage | None]) -> None:
        self.pages = pages

    def fetch_page(self, title: str) -> WikiPage | None:
        return self.pages.get(title)


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------


class TestPrompts(unittest.TestCase):
    def test_build_content_prompt_structure(self):
        system, user = build_content_prompt(
            title="Napoleon",
            year=1804,
            event_description="Napoleon crowned himself Emperor.",
            extract="Napoleon was a French general.",
            related=[("France", "France is a country.")],
        )
        self.assertIn("Napoleon", user)
        self.assertIn("France", user)
        self.assertIn("source_context", user)


class TestValidation(unittest.TestCase):
    def test_validate_suggestions_exact_match(self):
        available = ["France", "Waterloo", "Elba"]
        self.assertEqual(
            validate_suggestions(["France", "Elba"], available, max_suggestions=5),
            ["France", "Elba"],
        )

    def test_validate_suggestions_case_insensitive(self):
        available = ["France", "Waterloo"]
        self.assertEqual(
            validate_suggestions(["france", "WATERLOO"], available),
            ["France", "Waterloo"],
        )

    def test_validate_suggestions_underscore_normalisation(self):
        available = ["Saint Helena"]
        self.assertEqual(
            validate_suggestions(["Saint_Helena"], available),
            ["Saint Helena"],
        )

    def test_validate_suggestions_skips_unknown(self):
        available = ["France", "Waterloo"]
        self.assertEqual(
            validate_suggestions(["France", "Mars"], available),
            ["France"],
        )

    def test_validate_suggestions_respects_max(self):
        available = ["A", "B", "C", "D", "E", "F"]
        self.assertEqual(len(validate_suggestions(available, available, max_suggestions=3)), 3)

    def test_validate_suggestions_deduplicates(self):
        available = ["France"]
        self.assertEqual(
            validate_suggestions(["France", "france", "FRANCE"], available),
            ["France"],
        )


class TestJsonParsing(unittest.TestCase):
    def test_parse_llm_json_plain(self):
        raw = '{"source_context": "test", "reasoning": "r"}'
        self.assertEqual(_parse_llm_json(raw)["source_context"], "test")

    def test_parse_llm_json_with_markdown_fences(self):
        raw = '```json\n{"source_context": "test", "reasoning": "r"}\n```'
        self.assertEqual(_parse_llm_json(raw)["source_context"], "test")

    def test_parse_llm_json_missing_braces(self):
        with self.assertRaises(ValueError):
            _parse_llm_json("no json here")


class TestEnrichTopicHappyPath(unittest.TestCase):
    def test_enrich_topic_success(self):
        event = _event(
            "Napoleon became Emperor.",
            related=[_metric("Napoleon", 1000), _metric("France", 800), _metric("Waterloo", 600)],
            year=1804,
        )
        llm_response = {
            "source_context": "Napoleon crowned himself in 1804 after major victories.",
            "reasoning": "Combined main article with France context.",
        }
        fake_llm = FakeLLMClient(response=llm_response)
        fake_fetcher = FakeFetcher(
            pages={
                "Napoleon": _page("Napoleon", 1000),
                "France": _page("France", 800),
                "Waterloo": _page("Waterloo", 600),
            }
        )

        result = enrich_topic(event, fake_fetcher, fake_llm)

        self.assertIsInstance(result, EnrichedTopic)
        self.assertEqual(result.base_page.title, "Napoleon")
        self.assertEqual(len(result.related_pages), 2)
        self.assertEqual(result.source_context, llm_response["source_context"])
        self.assertIn("Combined", result.reasoning)
        self.assertEqual(result.combined_word_count, 2400)


class TestEnrichTopicFailSoft(unittest.TestCase):
    def test_enrich_topic_returns_fallback_on_llm_error(self):
        event = _event("Test event.", related=[_metric("Napoleon", 1000)])
        fake_llm = FakeLLMClient(raise_error=LLMError("API down"))
        fake_fetcher = FakeFetcher(
            pages={"Napoleon": _page("Napoleon", 1000)}
        )

        result = enrich_topic(event, fake_fetcher, fake_llm)

        self.assertEqual(result.base_page.title, "Napoleon")
        self.assertEqual(result.related_pages, [])
        self.assertEqual(result.source_context, "Test event.")
        self.assertIn("API down", result.reasoning)

    def test_enrich_topic_returns_fallback_on_invalid_json(self):
        event = _event("Test event.", related=[_metric("Napoleon", 1000)])

        class BadLLM:
            def chat(self, messages, temperature=0.3):
                return "not json"

        fake_fetcher = FakeFetcher(pages={"Napoleon": _page("Napoleon", 1000)})
        result = enrich_topic(event, fake_fetcher, BadLLM())
        self.assertEqual(result.related_pages, [])
        self.assertEqual(result.source_context, "Test event.")

    def test_enrich_topic_skips_missing_context_pages(self):
        event = _event(
            "Test.",
            related=[_metric("Napoleon", 1000), _metric("France", 800)],
        )
        llm_response = {
            "source_context": "Context.",
            "reasoning": "Used available sources.",
        }
        fake_llm = FakeLLMClient(response=llm_response)
        fake_fetcher = FakeFetcher(
            pages={"Napoleon": _page("Napoleon", 1000), "France": None}
        )

        result = enrich_topic(event, fake_fetcher, fake_llm)
        self.assertEqual(len(result.related_pages), 0)
        self.assertEqual(result.source_context, "Context.")

    def test_enrich_topic_skips_stub_context_pages(self):
        event = _event(
            "Test.",
            related=[_metric("Napoleon", 1000), _metric("Stubby", 50)],
        )
        llm_response = {
            "source_context": "Context.",
            "reasoning": "Skipped stub.",
        }
        fake_llm = FakeLLMClient(response=llm_response)
        fake_fetcher = FakeFetcher(
            pages={"Napoleon": _page("Napoleon", 1000), "Stubby": _page("Stubby", 50)}
        )

        result = enrich_topic(event, fake_fetcher, fake_llm)
        self.assertEqual(len(result.related_pages), 0)

    def test_enrich_topic_no_related_pages(self):
        event = _event("Fallback description.", related=[])
        fake_llm = FakeLLMClient(response={})
        fake_fetcher = FakeFetcher(pages={})

        result = enrich_topic(event, fake_fetcher, fake_llm)
        self.assertEqual(result.source_context, "Fallback description.")
        self.assertEqual(result.base_page.title, "Fallback description.")


class TestBatchEnrichment(unittest.TestCase):
    def test_enrich_topics_batch(self):
        events = [
            _event("Event A.", related=[_metric("A", 1000), _metric("B", 500)]),
            _event("Event C.", related=[_metric("C", 1000), _metric("D", 500)]),
        ]
        fake_llm = FakeLLMClient(
            response={"source_context": "Synthesized.", "reasoning": "r"}
        )
        fake_fetcher = FakeFetcher(
            pages={
                "A": _page("A", 1000),
                "B": _page("B", 500),
                "C": _page("C", 1000),
                "D": _page("D", 500),
            }
        )

        results = enrich_topics(events, fake_fetcher, fake_llm)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].base_page.title, "A")
        self.assertEqual(results[1].base_page.title, "C")
        self.assertEqual(results[0].source_context, "Synthesized.")


if __name__ == "__main__":
    unittest.main()
