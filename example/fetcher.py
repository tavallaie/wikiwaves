"""
WikiWaves Fetcher + Curator — Example Runner
=============================================
Fetches live Wikipedia data, enriches every related page,
and runs the curator to produce a scored, diverse selection.

Run with:
    uv run python example/fetcher.py
"""

from __future__ import annotations

import datetime
import json
import os
from dataclasses import asdict

from dotenv import load_dotenv
from loguru import logger

from wikiwaves.curator import aggregate_events, curate
from wikiwaves.fetcher import WikiFetcher
from wikiwaves.fetcher.exceptions import FetcherError

OUTPUT_DIR = "output"
PREFIX = "fetcher_"


def _save(name: str, data) -> str:
    """Save *data* to output/{prefix}{name}.json and return the path."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"{PREFIX}{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"Saved → {path}")
    return path


def _serialize_events(events):
    return [asdict(e) if hasattr(e, "__dataclass_fields__") else e for e in events]


def _serialize_edits(edits):
    return [asdict(e) if hasattr(e, "__dataclass_fields__") else e for e in edits]


def _serialize_pages(pages):
    return {
        k: (asdict(v) if v and hasattr(v, "__dataclass_fields__") else v)
        for k, v in pages.items()
    }


def _serialize_metrics(metrics):
    return {
        k: (asdict(v) if v and hasattr(v, "__dataclass_fields__") else v)
        for k, v in metrics.items()
    }


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
                    asdict(p) if hasattr(p, "__dataclass_fields__") else p
                    for p in evt.related_pages
                ],
            }
        )
    return result


def main() -> None:
    load_dotenv()

    try:
        fetcher = WikiFetcher()
    except RuntimeError as exc:
        logger.error(f"Cannot initialise fetcher: {exc}")
        raise SystemExit(1)

    today = datetime.date.today()
    logger.info(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")

    # ------------------------------------------------------------------ #
    # 1. On This Day
    # ------------------------------------------------------------------ #
    logger.info("STEP 1: fetch_on_this_day()")
    try:
        events = fetcher.fetch_on_this_day(today)
        _save("01_onthisday_events", _serialize_events(events))
    except FetcherError as exc:
        logger.error(f"STEP 1 failed: {exc}")

    # ------------------------------------------------------------------ #
    # 2. Recent Changes
    # ------------------------------------------------------------------ #
    logger.info("STEP 2: fetch_recent_changes(hours=24)")
    try:
        edits = fetcher.fetch_recent_changes(hours=24, limit=500)
        _save("02_recent_changes", _serialize_edits(edits))
    except FetcherError as exc:
        logger.error(f"STEP 2 failed: {exc}")

    # ------------------------------------------------------------------ #
    # 3. Search
    # ------------------------------------------------------------------ #
    logger.info("STEP 3: search('artificial intelligence')")
    try:
        canonical = fetcher.search("artificial intelligence")
        _save(
            "03_search_result",
            {"query": "artificial intelligence", "result": canonical},
        )
    except FetcherError as exc:
        logger.error(f"STEP 3 failed: {exc}")
        canonical = None

    # ------------------------------------------------------------------ #
    # 4. Single Page Fetch
    # ------------------------------------------------------------------ #
    title = canonical or "Python (programming language)"
    logger.info(f"STEP 4: fetch_page('{title}')")
    try:
        page = fetcher.fetch_page(title)
        _save("04_single_page", asdict(page) if page else None)
    except FetcherError as exc:
        logger.error(f"STEP 4 failed: {exc}")
        page = None

    # ------------------------------------------------------------------ #
    # 5. Bulk Page Fetch
    # ------------------------------------------------------------------ #
    titles = ["Napoleon", "Waterloo", "ThisPageDoesNotExist12345"]
    logger.info(f"STEP 5: fetch_pages({titles})")
    try:
        pages = fetcher.fetch_pages(titles)
        _save("05_bulk_pages", _serialize_pages(pages))
    except FetcherError as exc:
        logger.error(f"STEP 5 failed: {exc}")

    # ------------------------------------------------------------------ #
    # 6. Page Views
    # ------------------------------------------------------------------ #
    logger.info(f"STEP 6: fetch_page_views('{title}', days=30)")
    try:
        views = fetcher.fetch_page_views(title, days=30)
        _save(
            "06_page_views", {"title": title, "days": 30, "average_daily_views": views}
        )
    except FetcherError as exc:
        logger.error(f"STEP 6 failed: {exc}")

    # ------------------------------------------------------------------ #
    # 7. Article Quality Check
    # ------------------------------------------------------------------ #
    logger.info(f"STEP 7: check_article_quality('{title}')")
    try:
        is_fa, is_ga = fetcher.check_article_quality(title)
        _save("07_article_quality", {"title": title, "featured": is_fa, "good": is_ga})
    except FetcherError as exc:
        logger.error(f"STEP 7 failed: {exc}")

    # ------------------------------------------------------------------ #
    # 8. Enrich ALL related pages from On This Day
    # ------------------------------------------------------------------ #
    logger.info("STEP 8: enrich_pages() — all related pages")
    try:
        unique_titles: list[str] = []
        seen: set[str] = set()
        for evt in events:
            rel = getattr(evt, "related_titles", [])
            for t in rel:
                if t and t not in seen:
                    seen.add(t)
                    unique_titles.append(t)

        logger.info(f"Found {len(unique_titles)} unique page(s)")
        if unique_titles:
            enriched = fetcher.enrich_pages(unique_titles, days=30)
            _save("08_enriched_pages", _serialize_metrics(enriched))
        else:
            logger.warning("No related titles to enrich.")
            enriched = {}
    except FetcherError as exc:
        logger.error(f"STEP 8 failed: {exc}")
        enriched = {}

    # ------------------------------------------------------------------ #
    # 9. Curator — aggregate + score + select topics + fun facts
    # ------------------------------------------------------------------ #
    logger.info("STEP 9: curator — aggregate, score, select topics")
    try:
        # 9a. Aggregate events with enriched metrics
        aggregated = aggregate_events(events, enriched)
        _save("09_aggregated_events", _serialize_aggregated(aggregated))

        # 9b. Run full curation pipeline
        episode = curate(
            aggregated,
            date=today.isoformat(),
            topic_count=5,
        )

        curated_output = {
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
        }
        _save("09_curated_episode", curated_output)
    except Exception as exc:
        logger.error(f"STEP 9 failed: {exc}")

    # ------------------------------------------------------------------ #
    # 10. Summary
    # ------------------------------------------------------------------ #
    logger.info("STEP 10: summary")
    summary = {
        "date": today.isoformat(),
        "event_count": len(events) if "events" in dir() else 0,
        "edit_count": len(edits) if "edits" in dir() else 0,
        "sample_page": {
            "title": page.title if page else None,
            "word_count": page.word_count if page else None,
            "is_featured": page.is_featured if page else None,
            "is_good": page.is_good if page else None,
        },
    }
    _save("10_summary", summary)

    logger.info(f"\n✅ All done. Check the `{OUTPUT_DIR}/` folder.")


if __name__ == "__main__":
    main()
