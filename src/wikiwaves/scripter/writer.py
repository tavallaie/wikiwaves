"""Core script-writing logic — monologue format."""

from __future__ import annotations

import time

from loguru import logger

from wikiwaves.llm import LLMClient, LLMError

from .models import ScriptChunk, TopicScript
from .prompts import build_intro_prompt, build_script_prompt, build_transition_prompt


def _text_to_chunks(text: str, speaker: str = "Host") -> list[ScriptChunk]:
    """Split text into chunks by sentence for storage compatibility."""
    import re

    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [
        ScriptChunk(
            speaker=speaker,
            segment="main_event",
            text=s.strip(),
            emotion="neutral",
        )
        for s in sentences if s.strip()
    ]


def write_topic_script(
    title: str,
    year: int | None,
    event_description: str,
    source_context: str,
    llm_client: LLMClient,
) -> TopicScript:
    """Generate a monologue script for a single topic."""
    system_msg, user_msg = build_script_prompt(
        title=title,
        year=year,
        event_description=event_description,
        source_context=source_context,
    )

    try:
        raw = llm_client.chat(
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.4,
            max_tokens=25000,
        )
    except LLMError as exc:
        logger.warning(f"LLM failed for '{title}': {exc}")
        return TopicScript(
            topic_title=title,
            year=year,
            event_description=event_description,
            chunks=[],
        )

    text = raw.strip()
    chunks = _text_to_chunks(text)
    words = len(text.split())
    duration = words / 150.0

    logger.info(f"Script for '{title}': {words} words, ~{duration:.1f} min")

    return TopicScript(
        topic_title=title,
        year=year,
        event_description=event_description,
        chunks=chunks,
        estimated_duration_minutes=round(duration, 1),
    )


def write_transition(
    prev_title: str,
    prev_year: int | None,
    prev_description: str,
    next_title: str,
    next_year: int | None,
    next_description: str,
    llm_client: LLMClient | None = None,
) -> TopicScript:
    """Generate a one-sentence transition."""
    if llm_client is None:
        llm_client = LLMClient()

    system_msg, user_msg = build_transition_prompt(
        prev_title=prev_title,
        prev_year=prev_year,
        prev_description=prev_description,
        next_title=next_title,
        next_year=next_year,
        next_description=next_description,
    )

    try:
        raw = llm_client.chat(
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.4,
            max_tokens=25000,
        )
    except LLMError as exc:
        logger.warning(f"LLM failed for transition: {exc}")
        return TopicScript(
            topic_title=f"Transition: {prev_title} → {next_title}",
            year=None,
            event_description="",
            chunks=[],
        )

    text = raw.strip()
    chunks = _text_to_chunks(text)
    words = len(text.split())
    duration = words / 150.0

    logger.info(f"Transition: {words} words, ~{duration:.1f} min")

    return TopicScript(
        topic_title=f"Transition: {prev_title} → {next_title}",
        year=None,
        event_description="",
        chunks=chunks,
        estimated_duration_minutes=round(duration, 1),
    )


def write_episode_intro(
    date: str,
    topics: list[dict[str, object]],
    llm_client: LLMClient | None = None,
) -> TopicScript:
    """Generate an intro monologue previewing all topics."""
    if llm_client is None:
        llm_client = LLMClient()

    system_msg, user_msg = build_intro_prompt(date=date, topics=topics)

    try:
        raw = llm_client.chat(
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.4,
            max_tokens=25000,
        )
    except LLMError as exc:
        logger.warning(f"LLM failed for intro: {exc}")
        return TopicScript(
            topic_title="Episode Intro",
            year=None,
            event_description="",
            chunks=[],
        )

    text = raw.strip()
    chunks = _text_to_chunks(text)
    words = len(text.split())
    duration = words / 150.0

    logger.info(f"Intro: {words} words, ~{duration:.1f} min")

    return TopicScript(
        topic_title="Episode Intro",
        year=None,
        event_description="",
        chunks=chunks,
        estimated_duration_minutes=round(duration, 1),
    )
