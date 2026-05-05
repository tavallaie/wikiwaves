"""
WikiWaves Enricher — Example Runner
====================================
Takes the curated episode from example/curator.py, fetches full base pages,
and runs the LLM enricher to gather related articles for each topic.

Prerequisites:
    1. uv run python example/fetcher.py
    2. uv run python example/curator.py

Run:
    uv run python example/enricher.py
"""

from __future__ import annotations

import json
import os
import sys

from dataclasses import asdict
from dotenv import load_dotenv
from loguru import logger

from wikiwaves.enricher import EnrichedTopic, LLMClient, enrich_pages
from wikiwaves.fetcher import WikiFetcher

OUTPUT_DIR = "output"
PREFIX = "enricher_"


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


def _serialize_enriched(topics: list[EnrichedTopic]) -> list[dict]:
    """Convert EnrichedTopic objects to plain dicts for JSON."""
    return [
        {
            "base_page": asdict(t.base_page),
            "related_pages": [asdict(p) for p in t.related_pages],
            "reasoning": t.reasoning,
            "combined_word_count": t.combined_word_count,
        }
        for t in topics
    ]


def main() -> None:
    load_dotenv()

    curated_path = "output/curator_02_curated_episode.json"
    if not os.path.exists(curated_path):
        logger.error(f"Missing {curated_path} — run example/curator.py first")
        sys.exit(1)

    try:
        fetcher = WikiFetcher()
    except RuntimeError as exc:
        logger.error(f"Cannot initialise fetcher: {exc}")
        raise SystemExit(1)

    # Optional: LLMClient will read LLM_API_KEY etc. from env.
    # If no key is configured the enricher falls back to base-only topics.
    try:
        llm_client = LLMClient()
    except Exception as exc:
        logger.warning(f"LLM client not available ({exc}), enrichment will use base pages only")
        llm_client = None

    logger.info("Loading curated episode...")
    curated = _load_json(curated_path)
    topics = curated.get("topics", [])

    if not topics:
        logger.warning("No topics found in curated episode.")
        sys.exit(0)

    # Pick the first related page of each topic as the base article
    base_titles: list[str] = []
    for topic in topics:
        related = topic.get("related_pages", [])
        if related:
            base_titles.append(related[0]["title"])

    logger.info(f"Fetching full base pages for {len(base_titles)} topic(s)...")
    base_pages_map = fetcher.fetch_pages(base_titles)
    base_pages = [p for p in (base_pages_map.get(t) for t in base_titles) if p is not None]

    if not base_pages:
        logger.error("Could not fetch any base pages.")
        sys.exit(1)

    logger.info("Running enricher...")
    enriched = enrich_pages(base_pages, fetcher, llm_client)

    _save("01_enriched_topics", _serialize_enriched(enriched))

    print(f"\n{'=' * 60}")
    print("ENRICHED TOPICS")
    print(f"{'=' * 60}")
    for i, topic in enumerate(enriched, 1):
        print(
            f"\n{i}. {topic.base_page.title} "
            f"({topic.base_page.word_count} words)"
        )
        if topic.reasoning:
            print(f"   Reasoning: {topic.reasoning}")
        if topic.related_pages:
            print(f"   Related articles:")
            for p in topic.related_pages:
                print(f"      • {p.title} ({p.word_count} words)")
        else:
            print("   (no related articles)")

    print(f"\n{'=' * 60}")
    print("✅ Enricher example complete.")


if __name__ == "__main__":
    main()
