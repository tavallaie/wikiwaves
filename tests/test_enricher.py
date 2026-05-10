"""Tests for the WikiWaves Enricher module."""

from __future__ import annotations

import json
import unittest

from wikiwaves.enricher import (
    EnrichedTopic,
    enrich_page,
    enrich_pages,
    validate_suggestions,
)
from wikiwaves.llm import LLMClient, LLMError
from wikiwaves.enricher.expander import _parse_llm_json
from wikiwaves.enricher.prompts import build_link_suggestion_prompt
from wikiwaves.fetcher.models import WikiPage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _page(title: str, word_count: int = 500, links: list[str] | None = None) -> WikiPage:
    return WikiPage(
        title=title,
        plain_text=f"Extract for {title}",
        internal_links=links or [],
        word_count=word_count,
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

    def fetch_pages(self, titles: list[str]) -> dict[str, WikiPage | None]:
        return {t: self.pages.get(t) for t in titles}


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------


class TestPrompts(unittest.TestCase):
    def test_build_link_suggestion_prompt_structure(self):
        system, user = build_link_suggestion_prompt(
            title="Napoleon",
            extract="Napoleon was a French general.",
            links=["France", "Waterloo", "Elba"],
        )
        self.assertIn("Napoleon", user)
        self.assertIn("France", user)
        self.assertIn("Waterloo", user)


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
        raw = '{"suggestions": [{"title": "A"}], "explanation": "test"}'
        self.assertEqual(_parse_llm_json(raw)["suggestions"][0]["title"], "A")

    def test_parse_llm_json_with_markdown_fences(self):
        raw = '```json\n{"suggestions": [{"title": "A"}], "explanation": "test"}\n```'
        self.assertEqual(_parse_llm_json(raw)["suggestions"][0]["title"], "A")

    def test_parse_llm_json_missing_braces(self):
        with self.assertRaises(ValueError):
            _parse_llm_json("no json here")


class TestEnrichPageHappyPath(unittest.TestCase):
    def test_enrich_page_success(self):
        base = _page("Napoleon", word_count=1000, links=["France", "Waterloo", "Elba"])
        llm_response = {
            "suggestions": [
                {"title": "Waterloo", "reason": "Final battle"},
                {"title": "Elba", "reason": "Exile location"},
            ],
            "explanation": "Adds military and exile context.",
        }
        fake_llm = FakeLLMClient(response=llm_response)
        fake_fetcher = FakeFetcher(
            pages={
                "Waterloo": _page("Waterloo", word_count=800),
                "Elba": _page("Elba", word_count=600),
            }
        )

        result = enrich_page(base, fake_fetcher, fake_llm)

        self.assertIsInstance(result, EnrichedTopic)
        self.assertEqual(result.base_page.title, "Napoleon")
        self.assertEqual(len(result.related_pages), 2)
        self.assertEqual(result.combined_word_count, 2400)
        self.assertIn("military", result.reasoning.lower())


class TestEnrichPageFailSoft(unittest.TestCase):
    def test_enrich_page_returns_base_on_llm_error(self):
        base = _page("Napoleon", word_count=1000, links=["France"])
        fake_llm = FakeLLMClient(raise_error=LLMError("API down"))
        fake_fetcher = FakeFetcher(pages={})

        result = enrich_page(base, fake_fetcher, fake_llm)

        self.assertEqual(result.base_page.title, "Napoleon")
        self.assertEqual(result.related_pages, [])
        self.assertEqual(result.combined_word_count, 1000)

    def test_enrich_page_returns_base_on_invalid_json(self):
        base = _page("Napoleon", word_count=1000, links=["France"])

        class BadLLM:
            def chat(self, messages, temperature=0.3):
                return "not json"

        fake_fetcher = FakeFetcher(pages={})
        result = enrich_page(base, fake_fetcher, BadLLM())
        self.assertEqual(result.related_pages, [])

    def test_enrich_page_skips_missing_pages(self):
        base = _page("Napoleon", word_count=1000, links=["France", "Waterloo"])
        llm_response = {
            "suggestions": [
                {"title": "France", "reason": "Country"},
                {"title": "Waterloo", "reason": "Battle"},
            ],
            "explanation": "Context.",
        }
        fake_llm = FakeLLMClient(response=llm_response)
        fake_fetcher = FakeFetcher(
            pages={"France": _page("France", word_count=500), "Waterloo": None}
        )

        result = enrich_page(base, fake_fetcher, fake_llm)
        self.assertEqual(len(result.related_pages), 1)
        self.assertEqual(result.related_pages[0].title, "France")

    def test_enrich_page_skips_stubs(self):
        base = _page("Napoleon", word_count=1000, links=["France", "Stubby"])
        llm_response = {
            "suggestions": [
                {"title": "France", "reason": "Country"},
                {"title": "Stubby", "reason": "Tiny"},
            ],
            "explanation": "Context.",
        }
        fake_llm = FakeLLMClient(response=llm_response)
        fake_fetcher = FakeFetcher(
            pages={
                "France": _page("France", word_count=500),
                "Stubby": _page("Stubby", word_count=50),
            }
        )

        result = enrich_page(base, fake_fetcher, fake_llm)
        self.assertEqual(len(result.related_pages), 1)
        self.assertEqual(result.related_pages[0].title, "France")

    def test_enrich_page_no_valid_suggestions(self):
        base = _page("Napoleon", word_count=1000, links=["France"])
        llm_response = {
            "suggestions": [{"title": "Mars", "reason": "Planet"}],
            "explanation": "None valid.",
        }
        fake_llm = FakeLLMClient(response=llm_response)
        fake_fetcher = FakeFetcher(pages={})

        result = enrich_page(base, fake_fetcher, fake_llm)
        self.assertEqual(result.related_pages, [])
        self.assertIn("None valid.", result.reasoning)


class TestBatchEnrichment(unittest.TestCase):
    def test_enrich_pages_batch(self):
        pages = [
            _page("A", word_count=1000, links=["B"]),
            _page("C", word_count=1000, links=["D"]),
        ]
        fake_llm = FakeLLMClient(
            response={"suggestions": [{"title": "B", "reason": "r"}], "explanation": "e"}
        )
        fake_fetcher = FakeFetcher(
            pages={"B": _page("B", word_count=500), "D": _page("D", word_count=500)}
        )

        results = enrich_pages(pages, fake_fetcher, fake_llm)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].base_page.title, "A")
        self.assertEqual(results[1].base_page.title, "C")


if __name__ == "__main__":
    unittest.main()
