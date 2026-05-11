"""Prompt templates for the Enricher LLM calls."""

from __future__ import annotations

_SYSTEM_PROMPT = (
    "You are a research assistant. Your job is to combine multiple Wikipedia sources "
    "into a single coherent research brief. Do NOT write a podcast script. "
    "Produce factual, well-structured source material that a script writer can use later."
)

_SUMMARY_SYSTEM = (
    "You are a research assistant. Summarize the given Wikipedia article into a "
    "thorough research brief. Capture all key facts, dates, names, and historical significance."
)

_SUMMARY_TEMPLATE = """\
Summarize the following Wikipedia article into a thorough research brief.

Article: {title}
{extract}

Instructions:
- Capture all important facts, dates, names, and historical significance.
- Write a comprehensive summary (15–25 sentences) covering all key points.
- Do NOT add commentary, opinions, or podcast-style flair.
- Write in plain, factual prose.

Summary:
"""

_SYNTHESIS_TEMPLATE = """\
Combine the following article summaries into ONE unified research brief.

Historical Event ({year}):
{event_description}

Article Summaries:
{summaries}

Instructions:
- Review all summaries above. They are all related to the event, but some may be more relevant than others.
- Focus on the summaries that provide the most useful context for understanding this event.
- Combine the most relevant sources into a single coherent narrative (not bullet points).
- Preserve key facts, dates, names, and causal relationships.
- Eliminate redundant information.
- Do NOT add commentary, opinions, or podcast-style flair.
- Write in plain, factual prose suitable for a script writer to adapt later.
- Write a comprehensive brief covering the full story: background, the event itself, and its aftermath.

Output format (exactly):
=== SOURCE_CONTEXT ===
[Your combined research brief here]

=== REASONING ===
[Brief note on which articles you found most relevant and why]
"""


def build_summary_prompt(title: str, extract: str) -> tuple[str, str]:
    """Return (system_message, user_message) for per-page summarization."""
    user = _SUMMARY_TEMPLATE.format(
        title=title,
        extract=extract,
    )
    return _SUMMARY_SYSTEM, user


def build_synthesis_prompt(
    year: int | None,
    event_description: str,
    summaries: list[tuple[str, str]],
) -> tuple[str, str]:
    """Return (system_message, user_message) for final synthesis LLM call."""
    summary_parts: list[str] = []
    for title, summary in summaries:
        summary_parts.append(f"--- {title} ---\n{summary}")
    summaries_text = "\n\n".join(summary_parts) if summary_parts else "(none)"

    user = _SYNTHESIS_TEMPLATE.format(
        year=year if year is not None else "Unknown year",
        event_description=event_description,
        summaries=summaries_text,
    )
    return _SYSTEM_PROMPT, user
