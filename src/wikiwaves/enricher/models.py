"""Data models for the Enricher module."""

from __future__ import annotations

from dataclasses import dataclass, field

from wikiwaves.fetcher.models import WikiPage


@dataclass(frozen=True, slots=True)
class EnrichedTopic:
    """A base Wikipedia article bundled with LLM-suggested related articles.

    Attributes:
        base_page: The primary Wikipedia article for the topic.
        related_pages: Validated related articles fetched during enrichment.
        reasoning: The LLM's explanation for why these related articles were chosen.
        combined_word_count: Total word count of base + related pages.
    """

    base_page: WikiPage
    related_pages: list[WikiPage] = field(default_factory=list)
    reasoning: str = ""
    combined_word_count: int = 0
