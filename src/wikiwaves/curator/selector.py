"""Topic scoring, diversity filtering, and fun-fact extraction."""

from loguru import logger

from .models import AggregatedEvent, CuratedEpisode, TopicScore


def score_event(event: AggregatedEvent) -> TopicScore:
    """Compute a composite score for a single candidate topic.

    Scoring weights (from plan.md):
    - Content Richness : 30%
    - Recency Bias     : 20%
    - Page Popularity  : 15%
    - Article Quality  : 15%
    - Diversity        : 20% (applied later as a post-filter)
    """
    pages = event.related_pages

    # ---- Content Richness (30%) -----------------------------------------
    total_words = sum(p.word_count for p in pages)
    total_links = sum(p.internal_links for p in pages)
    richness = (
        min(total_words / 5_000, 1.0) + min(total_links / 250, 1.0)
    ) / 2

    # ---- Recency (20%) --------------------------------------------------
    recency = 0.0
    if event.year is not None:
        if event.year >= 1924:
            recency = 1.0
        elif event.year >= 1824:
            recency = 0.7
        elif event.year >= 1500:
            recency = 0.4
        else:
            recency = 0.2

    # ---- Popularity (15%) -----------------------------------------------
    views = [p.page_views_30d for p in pages if p.page_views_30d is not None]
    popularity = (
        min(sum(views) / max(len(views), 1) / 5_000, 1.0) if views else 0.0
    )

    # ---- Quality (15%) --------------------------------------------------
    quality = 0.0
    if pages:
        featured = sum(1 for p in pages if p.is_featured)
        good = sum(1 for p in pages if p.is_good)
        quality = min(
            (featured * 1.0 + good * 0.5) / len(pages), 1.0
        )

    # ---- Total (diversity applied later) --------------------------------
    total = richness * 0.30 + recency * 0.20 + popularity * 0.15 + quality * 0.15

    return TopicScore(
        event=event,
        content_richness=richness,
        recency=recency,
        popularity=popularity,
        quality=quality,
        total_score=total,
    )


def _century(year: int | None) -> int:
    return year // 100 if year is not None else -1


def select_topics(events: list[AggregatedEvent], count: int = 5) -> list[AggregatedEvent]:
    """Pick the *count* highest-scoring topics while enforcing era diversity.

    The algorithm:
    1. Score every event.
    2. Sort by total score descending.
    3. Greedily pick topics, skipping if their century is already
       well-represented (≥ 2 picks from the same century).
    4. If the diversity filter leaves the selection short, back-fill
       with the next best candidates.
    """
    if not events:
        return []

    scores = [score_event(e) for e in events]
    scores.sort(key=lambda s: s.total_score, reverse=True)

    selected: list[AggregatedEvent] = []
    century_counts: dict[int, int] = {}

    for sc in scores:
        if len(selected) >= count:
            break
        c = _century(sc.event.year)
        if century_counts.get(c, 0) >= 2:
            continue
        century_counts[c] = century_counts.get(c, 0) + 1
        selected.append(sc.event)

    # Back-fill if we were too aggressive with diversity
    if len(selected) < count:
        for sc in scores:
            if sc.event not in selected:
                selected.append(sc.event)
            if len(selected) >= count:
                break

    logger.info(
        f"Selected {len(selected)} topics from {len(events)} candidates"
    )
    return selected


def curate(
    events: list[AggregatedEvent],
    date: str,
    topic_count: int = 5,
) -> CuratedEpisode:
    """Run the full curation pipeline.

    Args:
        events: Aggregated On-This-Day events.
        date: Episode date (ISO-8601).
        topic_count: How many topics to select.

    Returns:
        A :class:`CuratedEpisode` with topics and scoring.
    """
    topics = select_topics(events, count=topic_count)
    breakdown = [score_event(e) for e in topics]

    return CuratedEpisode(
        date=date,
        topics=topics,
        total_events_considered=len(events),
        scoring_breakdown=breakdown,
    )
