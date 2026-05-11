"""Orchestration logic for enriching a topic into combined source context."""

from __future__ import annotations

import re

from loguru import logger

from wikiwaves.curator.models import AggregatedEvent
from wikiwaves.fetcher import WikiFetcher
from wikiwaves.fetcher.models import WikiPage

from wikiwaves.llm import LLMClient, LLMError
from .models import EnrichedTopic
from .prompts import build_summary_prompt, build_synthesis_prompt

DEFAULT_MIN_WORD_COUNT = 100  # skip stubs


def _parse_delimited_output(raw: str) -> tuple[str, str]:
    """Extract SOURCE_CONTEXT and REASONING from delimited LLM output.

    Returns (source_context, reasoning). Falls back to the raw text as
    source_context if delimiters are missing.
    """
    ctx_match = re.search(
        r"===\s*SOURCE_CONTEXT\s*===\s*\n?(.*?)\n?(?:===\s*REASONING\s*===|$)",
        raw,
        re.DOTALL | re.IGNORECASE,
    )
    source_context = ctx_match.group(1).strip() if ctx_match else raw.strip()

    reason_match = re.search(
        r"===\s*REASONING\s*===\s*\n?(.*?)$",
        raw,
        re.DOTALL | re.IGNORECASE,
    )
    reasoning = reason_match.group(1).strip() if reason_match else ""

    return source_context, reasoning


def _summarize_page(page: WikiPage, llm_client: LLMClient) -> str:
    """Ask the LLM to summarize a single Wikipedia page."""
    system_msg, user_msg = build_summary_prompt(
        title=page.title,
        extract=page.plain_text,
    )
    raw = llm_client.chat(
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.3,
    )
    return raw.strip()


def enrich_topic(
    event: AggregatedEvent,
    fetcher: WikiFetcher,
    llm_client: LLMClient,
) -> EnrichedTopic:
    """Expand an *event* into combined source context.

    Two-stage LLM pipeline:
        1. Fetch ALL related pages (full extracts).
        2. Summarize each page individually via LLM.
        3. Feed event + all summaries to LLM for final synthesis.

    Fail-soft: if any step fails, returns an :class:`EnrichedTopic` containing
    only a placeholder base page and empty context.
    """
    if not event.related_pages:
        return EnrichedTopic(
            base_page=WikiPage(title=event.description[:50]),
            reasoning="No related pages available.",
            source_context=event.description,
        )

    try:
        # ------------------------------------------------------------------
        # Stage 0: Fetch ALL related pages with full content
        # ------------------------------------------------------------------
        all_pages: list[WikiPage] = []
        for page_metric in event.related_pages:
            page = fetcher.fetch_page(page_metric.title)
            if page is None:
                logger.warning(f"Related page '{page_metric.title}' does not exist")
                continue
            if page.word_count < DEFAULT_MIN_WORD_COUNT:
                logger.warning(
                    f"Skipping stub '{page_metric.title}' ({page.word_count} words)"
                )
                continue
            all_pages.append(page)
            logger.info(
                f"Fetched '{page.title}' — {page.word_count} words (full extract)"
            )

        if not all_pages:
            logger.warning("All related pages were missing or too short")
            return EnrichedTopic(
                base_page=WikiPage(title=event.description[:50]),
                reasoning="All related pages missing or too short.",
                source_context=event.description,
            )

        # Pick the longest as base_page (for file naming / backward compat)
        base_page = max(all_pages, key=lambda p: p.word_count)

        # ------------------------------------------------------------------
        # Stage 1: Summarize each page individually
        # ------------------------------------------------------------------
        page_summaries: dict[str, str] = {}
        for page in all_pages:
            try:
                summary = _summarize_page(page, llm_client)
                page_summaries[page.title] = summary
                logger.info(f"Summarized '{page.title}' — {len(summary)} chars")
            except LLMError as exc:
                logger.warning(f"Failed to summarize '{page.title}': {exc}")
                # Fall back to using the raw extract as its own "summary"
                page_summaries[page.title] = page.plain_text[:2000]

        # ------------------------------------------------------------------
        # Stage 2: Synthesize final content from summaries + event
        # ------------------------------------------------------------------
        summary_tuples = [
            (title, page_summaries[title])
            for title in page_summaries
        ]
        system_msg, user_msg = build_synthesis_prompt(
            year=event.year,
            event_description=event.description,
            summaries=summary_tuples,
        )

        raw_response = llm_client.chat(
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
        )

        source_context, reasoning = _parse_delimited_output(raw_response)

        total_words = sum(p.word_count for p in all_pages)

        return EnrichedTopic(
            base_page=base_page,
            related_pages=[p for p in all_pages if p.title != base_page.title],
            reasoning=reasoning,
            source_context=source_context,
            combined_word_count=total_words,
            page_summaries=page_summaries,
        )

    except (LLMError, ValueError, KeyError) as exc:
        logger.warning(f"Enrichment failed for event '{event.event_id}': {exc}")
        return EnrichedTopic(
            base_page=WikiPage(title=event.description[:50]),
            reasoning=f"LLM/synthesis failed: {exc}",
            source_context=event.description,
        )
    except Exception as exc:
        logger.error(f"Unexpected enrichment error for event '{event.event_id}': {exc}")
        return EnrichedTopic(
            base_page=WikiPage(title=event.description[:50]),
            reasoning=f"Unexpected error: {exc}",
            source_context=event.description,
        )


def enrich_topics(
    events: list[AggregatedEvent],
    fetcher: WikiFetcher,
    llm_client: LLMClient | None = None,
) -> list[EnrichedTopic]:
    """Enrich multiple topics into source context.

    Args:
        events: Curated topics to expand.
        fetcher: Instance of :class:`WikiFetcher`.
        llm_client: LLM client. If ``None``, a default :class:`LLMClient` is created.

    Returns:
        A list of :class:`EnrichedTopic` objects in the same order as *events*.
    """
    if llm_client is None:
        llm_client = LLMClient()

    results: list[EnrichedTopic] = []
    for event in events:
        enriched = enrich_topic(event, fetcher, llm_client)
        results.append(enriched)

    return results
