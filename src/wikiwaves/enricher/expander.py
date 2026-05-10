"""Orchestration logic for enriching a base page with related articles."""

from __future__ import annotations

import json
import re

from loguru import logger

from wikiwaves.fetcher import WikiFetcher
from wikiwaves.fetcher.models import WikiPage

from wikiwaves.llm import LLMClient, LLMError
from .models import EnrichedTopic
from .prompts import build_link_suggestion_prompt
from .validators import validate_suggestions

DEFAULT_MAX_RELATED = 5
DEFAULT_MIN_WORD_COUNT = 100  # skip stubs


def _parse_llm_json(raw: str) -> dict:
    """Extract a JSON object from an LLM response, stripping markdown fences."""
    cleaned = re.sub(r"^```(?:json)?\s*|```\s*$", "", raw.strip(), flags=re.MULTILINE)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in LLM response")
    return json.loads(cleaned[start : end + 1])


def enrich_page(
    base_page: WikiPage,
    fetcher: WikiFetcher,
    llm_client: LLMClient,
    max_related: int = DEFAULT_MAX_RELATED,
    min_word_count: int = DEFAULT_MIN_WORD_COUNT,
) -> EnrichedTopic:
    """Expand *base_page* with LLM-suggested related articles.

    Fail-soft: if any step fails, returns an :class:`EnrichedTopic` containing
    only the base page and empty related pages.
    """
    try:
        system_msg, user_msg = build_link_suggestion_prompt(
            title=base_page.title,
            extract=base_page.plain_text,
            links=base_page.internal_links,
        )

        raw_response = llm_client.chat(
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
        )

        data = _parse_llm_json(raw_response)
        raw_suggestions = [
            s["title"]
            for s in data.get("suggestions", [])
            if isinstance(s, dict) and "title" in s
        ]
        reasoning = data.get("explanation", "") or ""

        validated = validate_suggestions(
            raw_suggestions,
            base_page.internal_links,
            max_suggestions=max_related,
        )

        if not validated:
            logger.warning(f"No valid link suggestions for '{base_page.title}'")
            return EnrichedTopic(
                base_page=base_page,
                related_pages=[],
                reasoning=reasoning,
                combined_word_count=base_page.word_count,
            )

        fetched = fetcher.fetch_pages(validated)
        related: list[WikiPage] = []
        for title in validated:
            page = fetched.get(title)
            if page is None:
                logger.warning(f"Suggested page '{title}' does not exist")
                continue
            if page.word_count < min_word_count:
                logger.warning(f"Skipping stub '{title}' ({page.word_count} words)")
                continue
            related.append(page)

        total_words = base_page.word_count + sum(p.word_count for p in related)

        return EnrichedTopic(
            base_page=base_page,
            related_pages=related,
            reasoning=reasoning,
            combined_word_count=total_words,
        )

    except (LLMError, json.JSONDecodeError, ValueError, KeyError) as exc:
        logger.warning(f"Enrichment failed for '{base_page.title}': {exc}")
        return EnrichedTopic(
            base_page=base_page,
            related_pages=[],
            reasoning="",
            combined_word_count=base_page.word_count,
        )
    except Exception as exc:
        logger.error(f"Unexpected enrichment error for '{base_page.title}': {exc}")
        return EnrichedTopic(
            base_page=base_page,
            related_pages=[],
            reasoning="",
            combined_word_count=base_page.word_count,
        )


def enrich_pages(
    base_pages: list[WikiPage],
    fetcher: WikiFetcher,
    llm_client: LLMClient | None = None,
    max_related: int = DEFAULT_MAX_RELATED,
    min_word_count: int = DEFAULT_MIN_WORD_COUNT,
) -> list[EnrichedTopic]:
    """Enrich multiple base pages.

    Args:
        base_pages: Primary Wikipedia articles to expand.
        fetcher: Instance of :class:`WikiFetcher`.
        llm_client: LLM client. If ``None``, a default :class:`LLMClient` is created.
        max_related: Maximum related articles per topic.
        min_word_count: Minimum word count for a related page to be accepted.

    Returns:
        A list of :class:`EnrichedTopic` objects in the same order as *base_pages*.
    """
    if llm_client is None:
        llm_client = LLMClient()

    results: list[EnrichedTopic] = []
    for page in base_pages:
        enriched = enrich_page(
            page,
            fetcher,
            llm_client,
            max_related=max_related,
            min_word_count=min_word_count,
        )
        results.append(enriched)

    return results
