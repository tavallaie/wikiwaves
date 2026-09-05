"""Data models for podcast scripts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True, slots=True)
class ScriptChunk:
    """A single line of dialogue in the podcast script."""

    speaker: Literal["Host", "Expert"]
    segment: Literal["hook", "context", "main_event", "impact", "conclusion"]
    text: str
    emotion: Literal["neutral", "excited", "curious", "surprised", "somber"] = "neutral"


@dataclass(frozen=True, slots=True)
class TopicScript:
    """A complete script for one topic."""

    topic_title: str
    year: int | None
    event_description: str
    chunks: list[ScriptChunk] = field(default_factory=list)
    estimated_duration_minutes: float = 0.0


@dataclass(frozen=True, slots=True)
class EpisodeScript:
    """A complete episode script covering all topics."""

    episode_title: str
    date: str
    topics: list[TopicScript] = field(default_factory=list)
    intro_text: str = ""
    outro_text: str = ""
    total_duration_minutes: float = 0.0
