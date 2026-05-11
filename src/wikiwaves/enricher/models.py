"""Data models for the Enricher module."""

from __future__ import annotations

from dataclasses import dataclass, field

from wikiwaves.fetcher.models import WikiPage


@dataclass(frozen=True, slots=True)
class EnrichedTopic:
    """A topic enriched with combined source context for downstream script writing.

    Attributes:
        base_page: The primary Wikipedia article for the topic.
        related_pages: Additional articles fetched for context.
        reasoning: Note on why these sources were chosen.
        source_context: Combined source material (main + related articles + event)
            synthesized into a single coherent document for the script writer.
        combined_word_count: Total word count of base + related pages.
    """

    base_page: WikiPage
    related_pages: list[WikiPage] = field(default_factory=list)
    reasoning: str = ""
    source_context: str = ""
    combined_word_count: int = 0
    page_summaries: dict[str, str] = field(default_factory=dict)
