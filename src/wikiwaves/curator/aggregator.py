"""Aggregation logic: merge On-This-Day events with enriched page metrics."""

from wikiwaves.fetcher.models import OnThisDayEvent, PageMetrics

from .models import AggregatedEvent


def aggregate_events(
    events: list[OnThisDayEvent],
    enriched: dict[str, PageMetrics],
) -> list[AggregatedEvent]:
    """Bundle each On-This-Day event with its enriched related pages.

    Args:
        events: Raw events from :meth:`WikiFetcher.fetch_on_this_day`.
        enriched: Mapping from page title to :class:`PageMetrics`.

    Returns:
        A list of :class:`AggregatedEvent` objects ready for scoring.
    """
    result: list[AggregatedEvent] = []
    for event in events:
        related = []
        for title in event.related_titles:
            metrics = enriched.get(title)
            if metrics is not None:
                related.append(metrics)
        result.append(
            AggregatedEvent(
                event_id=event.event_id,
                year=event.year,
                description=event.description,
                event_type=event.event_type,
                related_pages=related,
            )
        )
    return result
