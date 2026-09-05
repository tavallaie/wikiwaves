"""Prompt templates for monologue podcast scripts."""

from __future__ import annotations

_SYSTEM_PROMPT = (
    "You are a podcast narrator. You tell historical stories in an engaging, "
    "conversational style. Use plain text only. No markdown."
)

_TOPIC_TEMPLATE = """\
Write a short podcast monologue about this historical topic.

Topic: {title}
Year: {year}
Event: {event_description}

Source Material:
{source_context}

Instructions:
- Write 8–12 sentences.
- Tell it like a story: set the scene, describe the event, explain why it matters.
- Use casual, conversational language. Short sentences.
- Do NOT list facts. Weave them into a narrative.
- Do NOT use markdown, **, *, or bullet points.
- Write in plain paragraphs.

Now write the monologue:
"""

_INTRO_TEMPLATE = """\
Write a short podcast intro previewing today's topics.

Date: {date}
Topics:
{topics}

Instructions:
- The host speaks directly to the audience.
- 3–5 sentences.
- Welcome to WikiWaves. Build excitement.
- Briefly mention each topic with its year.
- Plain text only. No markdown.

Now write the intro:
"""

_TRANSITION_TEMPLATE = """\
Write a one-sentence transition between two podcast segments.

Previous: {prev_title} ({prev_year})
Next: {next_title} ({next_year})

Instructions:
- One smooth sentence that bridges the previous topic to the next.
- Plain text only.

Now write the transition:
"""


def build_script_prompt(
    title: str,
    year: int | None,
    event_description: str,
    source_context: str,
) -> tuple[str, str]:
    """Return (system_message, user_message) for topic monologue."""
    user = _TOPIC_TEMPLATE.format(
        title=title,
        year=year if year is not None else "Unknown",
        event_description=event_description,
        source_context=source_context,
    )
    return _SYSTEM_PROMPT, user


def build_intro_prompt(date: str, topics: list[dict[str, object]]) -> tuple[str, str]:
    """Return (system_message, user_message) for episode intro."""
    topic_lines = []
    for i, t in enumerate(topics, 1):
        year = t.get("year", "Unknown")
        desc = t.get("event_description", "")
        topic_lines.append(f"{i}. ({year}) {desc}")
    topics_text = "\n".join(topic_lines)

    user = _INTRO_TEMPLATE.format(date=date, topics=topics_text)
    return _SYSTEM_PROMPT, user


def build_transition_prompt(
    prev_title: str,
    prev_year: int | None,
    prev_description: str,
    next_title: str,
    next_year: int | None,
    next_description: str,
) -> tuple[str, str]:
    """Return (system_message, user_message) for transition."""
    user = _TRANSITION_TEMPLATE.format(
        prev_title=prev_title,
        prev_year=prev_year if prev_year is not None else "Unknown",
        next_title=next_title,
        next_year=next_year if next_year is not None else "Unknown",
    )
    return _SYSTEM_PROMPT, user
