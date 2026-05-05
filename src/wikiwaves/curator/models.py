"""Data models for the Curator module."""

from dataclasses import dataclass, field

from wikiwaves.fetcher.models import PageMetrics


@dataclass(frozen=True, slots=True)
class AggregatedEvent:
    """An On-This-Day event bundled with enriched metrics for its related pages."""

    event_id: str
    year: int | None
    description: str
    event_type: str
    related_pages: list[PageMetrics] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class TopicScore:
    """Transparent scoring breakdown for a candidate topic."""

    event: AggregatedEvent
    content_richness: float = 0.0
    recency: float = 0.0
    popularity: float = 0.0
    quality: float = 0.0
    diversity_penalty: float = 0.0
    total_score: float = 0.0


@dataclass(frozen=True, slots=True)
class CuratedEpisode:
    """The final curated output for one day."""

    date: str
    topics: list[AggregatedEvent] = field(default_factory=list)
    total_events_considered: int = 0
    scoring_breakdown: list[TopicScore] = field(default_factory=list)
