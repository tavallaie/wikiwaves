"""Validation logic for LLM link suggestions."""

from __future__ import annotations


def validate_suggestions(
    suggestions: list[str],
    available_links: list[str],
    max_suggestions: int = 5,
) -> list[str]:
    """Return *suggestions* filtered to titles that actually exist in *available_links*.

    Matching is case-insensitive and ignores underscores vs spaces.
    """
    link_map: dict[str, str] = {}
    for link in available_links:
        key = link.lower().replace("_", " ")
        if key not in link_map:
            link_map[key] = link

    validated: list[str] = []
    for raw in suggestions:
        key = raw.lower().replace("_", " ")
        original = link_map.get(key)
        if original and original not in validated:
            validated.append(original)
        if len(validated) >= max_suggestions:
            break

    return validated
