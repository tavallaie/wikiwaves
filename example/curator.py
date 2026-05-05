"""
WikiWaves Curator — Standalone Example
========================================
Demonstrates the curator pipeline using data already saved in output/.

If you haven't run example/fetcher.py yet, do that first:
    uv run python example/fetcher.py

Then run this:
    uv run python example/curator.py
"""

from __future__ import annotations

import json
import os
import sys

from loguru import logger

from wikiwaves.curator import aggregate_events, curate
from wikiwaves.fetcher.models import OnThisDayEvent, PageMetrics

OUTPUT_DIR = "output"
PREFIX = "curator_"


def _save(name: str, data) -> str:
    """Save *data* to output/{prefix}{name}.json and return the path."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"{PREFIX}{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"Saved → {path}")
    return path


def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _reconstruct_events(raw: list[dict]) -> list[OnThisDayEvent]:
    return [
        OnThisDayEvent(
            event_id=item.get("event_id", ""),
            year=item.get("year"),
            description=item.get("description", ""),
            related_titles=item.get("related_titles", []),
            event_type=item.get("event_type", "selected"),
        )
        for item in raw
    ]


def _reconstruct_metrics(raw: dict[str, dict | None]) -> dict[str, PageMetrics]:
    result: dict[str, PageMetrics] = {}
    for title, data in raw.items():
        if data is None:
            continue
        result[title] = PageMetrics(
            title=data.get("title", title),
            word_count=data.get("word_count", 0),
            internal_links=data.get("internal_links", 0),
            external_links=data.get("external_links", 0),
            page_views_30d=data.get("page_views_30d"),
            edit_count_30d=data.get("edit_count_30d"),
            is_featured=data.get("is_featured", False),
            is_good=data.get("is_good", False),
            source_url=data.get("source_url", ""),
        )
    return result


def _serialize_aggregated(aggregated):
    """Convert AggregatedEvent list to plain dicts for JSON."""
    result = []
    for evt in aggregated:
        result.append(
            {
                "event_id": evt.event_id,
                "year": evt.year,
                "description": evt.description,
                "event_type": evt.event_type,
                "related_pages": [
                    {
                        "title": p.title,
                        "word_count": p.word_count,
                        "internal_links": p.internal_links,
                        "external_links": p.external_links,
                        "page_views_30d": p.page_views_30d,
                        "edit_count_30d": p.edit_count_30d,
                        "is_featured": p.is_featured,
                        "is_good": p.is_good,
                        "source_url": p.source_url,
                    }
                    for p in evt.related_pages
                ],
            }
        )
    return result


def main() -> None:
    # ------------------------------------------------------------------ #
    # Load fetcher output
    # ------------------------------------------------------------------ #
    events_path = "output/fetcher_01_onthisday_events.json"
    enriched_path = "output/fetcher_08_enriched_pages.json"

    for p in (events_path, enriched_path):
        if not os.path.exists(p):
            logger.error(f"Missing {p} — run 'uv run python example/fetcher.py' first")
            sys.exit(1)

    logger.info("Loading fetcher output...")
    raw_events = _load_json(events_path)
    raw_enriched = _load_json(enriched_path)

    events = _reconstruct_events(raw_events)
    enriched = _reconstruct_metrics(raw_enriched)

    logger.info(f"Loaded {len(events)} events, {len(enriched)} enriched pages")

    # ------------------------------------------------------------------ #
    # Step 1: Aggregate
    # ------------------------------------------------------------------ #
    logger.info("STEP 1: aggregate_events()")
    aggregated = aggregate_events(events, enriched)
    logger.info(f"Aggregated {len(aggregated)} events")

    # Save aggregated output
    _save("01_aggregated_events", _serialize_aggregated(aggregated))

    # Show a preview
    for evt in aggregated[:3]:
        total_views = sum((p.page_views_30d or 0) for p in evt.related_pages)
        total_edits = sum((p.edit_count_30d or 0) for p in evt.related_pages)
        print(
            f"\n  [{evt.event_type}] Year={evt.year}\n"
            f"  {evt.description[:80]}...\n"
            f"  Related pages: {len(evt.related_pages)}  "
            f"(combined views: {total_views:,}, edits: {total_edits})"
        )
    if len(aggregated) > 3:
        print(f"  ... and {len(aggregated) - 3} more events")

    # ------------------------------------------------------------------ #
    # Step 2: Curate
    # ------------------------------------------------------------------ #
    logger.info("\nSTEP 2: curate()")
    episode = curate(
        aggregated,
        date="2026-05-05",
        topic_count=5,
    )

    # Save curated episode
    _save("02_curated_episode", {
        "date": episode.date,
        "topics": _serialize_aggregated(episode.topics),
        "total_events_considered": episode.total_events_considered,
        "scoring_breakdown": [
            {
                "event_description": sb.event.description[:80],
                "content_richness": sb.content_richness,
                "recency": sb.recency,
                "popularity": sb.popularity,
                "quality": sb.quality,
                "diversity_penalty": sb.diversity_penalty,
                "total_score": sb.total_score,
            }
            for sb in episode.scoring_breakdown
        ],
    })

    print(f"\n{'=' * 60}")
    print("CURATED EPISODE")
    print(f"{'=' * 60}")
    print(f"Date: {episode.date}")
    print(f"Events considered: {episode.total_events_considered}")
    print(f"Topics selected: {len(episode.topics)}")

    print("\n--- Selected Topics ---")
    for i, topic in enumerate(episode.topics, 1):
        print(
            f"\n{i}. [{topic.year}] {topic.description[:70]}...\n"
            f"   Related pages: {len(topic.related_pages)}"
        )
        for p in topic.related_pages:
            print(
                f"      • {p.title}: "
                f"{p.page_views_30d or 0:,} views, "
                f"{p.edit_count_30d or 0} edits, "
                f"{p.internal_links} internal links"
            )

    print(f"\n--- Scoring Breakdown ---")
    for sb in episode.scoring_breakdown:
        print(
            f"  {sb.event.description[:50]}...  "
            f"score={sb.total_score:.3f}  "
            f"(richness={sb.content_richness:.2f}  "
            f"recency={sb.recency:.2f}  "
            f"popularity={sb.popularity:.2f}  "
            f"quality={sb.quality:.2f})"
        )

    print(f"\n{'=' * 60}")
    print("✅ Curator example complete.")


if __name__ == "__main__":
    main()
