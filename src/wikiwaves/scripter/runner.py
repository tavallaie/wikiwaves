"""Standalone script writer — reads enriched topic .txt files and generates monologue scripts.

Usage:
    uv run python -m wikiwaves.scripter.runner
    uv run python -m wikiwaves.scripter.runner 2026-05-11
"""

from __future__ import annotations

import argparse
import datetime
import json
import os

from loguru import logger

from wikiwaves.llm import LLMClient
from wikiwaves.scripter import write_episode_intro, write_topic_script, write_transition


def _parse_topic_file(path: str) -> dict[str, str | int | None]:
    """Parse a topic_*.txt file into title, year, event, source_context."""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    lines = content.splitlines()
    title = ""
    year = None
    event = ""

    in_header = True
    header_lines = []
    body_lines = []

    for line in lines:
        if in_header:
            if line.startswith("-" * 40):
                in_header = False
                continue
            header_lines.append(line)
        else:
            body_lines.append(line)

    for h in header_lines:
        if h.startswith("Topic: "):
            title = h[7:].strip()
        elif h.startswith("Year: "):
            try:
                year = int(h[6:].strip())
            except ValueError:
                year = None
        elif h.startswith("Event: "):
            event = h[7:].strip()

    source_context = "\n".join(body_lines).strip()

    return {
        "title": title,
        "year": year,
        "event_description": event,
        "source_context": source_context,
    }


def run(date_str: str | None = None, output_dir: str = "output") -> None:
    """Generate monologue scripts from existing topic .txt files."""
    date_str = date_str or datetime.date.today().isoformat()
    day_dir = os.path.join(output_dir, date_str)

    if not os.path.isdir(day_dir):
        logger.error(f"Directory not found: {day_dir}")
        return

    topic_files = sorted(
        f for f in os.listdir(day_dir) if f.startswith("topic_") and f.endswith(".txt")
    )
    if not topic_files:
        logger.error(f"No topic_*.txt files found in {day_dir}")
        return

    topics: list[dict[str, str | int | None]] = []
    for fname in topic_files:
        path = os.path.join(day_dir, fname)
        topic = _parse_topic_file(path)
        topics.append(topic)
        logger.info(f"Parsed {fname} — {topic['title']} ({topic['year']})")

    llm_client = LLMClient()

    # Generate intro
    logger.info("Generating intro...")
    intro = write_episode_intro(
        date=date_str,
        topics=[
            {"title": t["title"], "year": t["year"], "event_description": t["event_description"]}
            for t in topics
        ],
        llm_client=llm_client,
    )
    intro_path = os.path.join(day_dir, "script_00_intro.txt")
    with open(intro_path, "w", encoding="utf-8") as f:
        f.write(f"{intro.topic_title}\n")
        f.write(f"Duration: ~{intro.estimated_duration_minutes} min\n\n")
        for c in intro.chunks:
            f.write(f"{c.text}\n")
    logger.info(f"Saved intro → {intro_path}")

    all_scripts: list[dict[str, object]] = [
        {
            "type": "intro",
            "title": intro.topic_title,
            "text": " ".join(c.text for c in intro.chunks),
        }
    ]

    # Generate topic scripts and transitions
    for i, topic in enumerate(topics):
        logger.info(f"Generating script for {topic['title']}...")
        script = write_topic_script(
            title=topic["title"],
            year=topic["year"],
            event_description=topic["event_description"],
            source_context=topic["source_context"],
            llm_client=llm_client,
        )

        safe_title = str(topic["title"]).replace(" ", "_").replace("/", "-")
        script_path = os.path.join(day_dir, f"script_{i + 1:02d}_{safe_title}.txt")
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(f"{script.topic_title} ({script.year})\n")
            f.write(f"Duration: ~{script.estimated_duration_minutes} min\n\n")
            for c in script.chunks:
                f.write(f"{c.text}\n")
        logger.info(f"Saved script → {script_path}")

        all_scripts.append({
            "type": "topic",
            "title": script.topic_title,
            "year": script.year,
            "text": " ".join(c.text for c in script.chunks),
        })

        # Transition to next topic
        if i < len(topics) - 1:
            next_topic = topics[i + 1]
            logger.info(f"Generating transition to {next_topic['title']}...")
            transition = write_transition(
                prev_title=str(topic["title"]),
                prev_year=topic["year"],
                prev_description=str(topic["event_description"]),
                next_title=str(next_topic["title"]),
                next_year=next_topic["year"],
                next_description=str(next_topic["event_description"]),
                llm_client=llm_client,
            )

            trans_path = os.path.join(day_dir, f"script_{i + 1:02d}_to_{i + 2:02d}_transition.txt")
            with open(trans_path, "w", encoding="utf-8") as f:
                f.write(f"Transition\n")
                f.write(f"Duration: ~{transition.estimated_duration_minutes} min\n\n")
                for c in transition.chunks:
                    f.write(f"{c.text}\n")
            logger.info(f"Saved transition → {trans_path}")

            all_scripts.append({
                "type": "transition",
                "title": transition.topic_title,
                "text": " ".join(c.text for c in transition.chunks),
            })

    # Save combined JSON for reference
    combined_path = os.path.join(day_dir, "scripts.json")
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_scripts, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved combined scripts → {combined_path}")
    logger.info(f"Done. Generated {len(topics)} topic scripts + intro + transitions.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate monologue podcast scripts.")
    parser.add_argument("date", nargs="?", help="Date folder (YYYY-MM-DD). Defaults to today.")
    parser.add_argument("--output-dir", default="output", help="Output directory.")
    args = parser.parse_args()
    run(date_str=args.date, output_dir=args.output_dir)
