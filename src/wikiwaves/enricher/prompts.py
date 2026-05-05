"""Prompt templates for the Enricher LLM calls."""

from __future__ import annotations

_SYSTEM_PROMPT = (
    "You are a research assistant helping to prepare content for a podcast episode. "
    "Your task is to select Wikipedia articles that add the most valuable context "
    "to a main topic. Return only valid JSON."
)

_USER_TEMPLATE = """\
Given the Wikipedia article "{title}" and the following extract, select 3 to 5 internal links that would provide the most valuable additional context for listeners.

Choose links that explain:
- Causes or background
- Consequences or aftermath
- Cultural impact or legacy

Avoid:
- Lists or timelines
- Disambiguation pages
- Overly broad topics (e.g., "United States", "Earth")
- Years or dates as standalone articles

Return ONLY a JSON object with this exact structure:
{{
  "suggestions": [
    {{"title": "Article Title", "reason": "Brief reason for selection"}}
  ],
  "explanation": "One-sentence summary of why these links were chosen."
}}

Article extract:
{extract}

Available internal links ({count} total):
{links}
"""


def build_link_suggestion_prompt(title: str, extract: str, links: list[str]) -> tuple[str, str]:
    """Return (system_message, user_message) for the link-suggestion LLM call."""
    truncated = extract[:2000] if extract else ""
    links_text = "\n".join(f"- {link}" for link in links[:200])
    user = _USER_TEMPLATE.format(
        title=title,
        extract=truncated,
        count=len(links),
        links=links_text,
    )
    return _SYSTEM_PROMPT, user
