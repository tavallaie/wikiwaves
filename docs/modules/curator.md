# Curator — Topic Selection

## Purpose
Transform a large pool of raw events into a tight, diverse, high-quality selection of topics for an episode. Extract the most interesting community activity as "fun facts."

## Input
- Raw "On This Day" events
- Recent edits / community activity
- Optional article metadata (length, quality, popularity)

## Output
- A fixed number of selected topics (e.g., 5)
- A short list of fun facts
- Scoring rationale for each selection

## Boundaries
- Applies transparent, configurable scoring heuristics.
- Enforces diversity across eras, themes, or categories.
- Avoids stubs or articles with insufficient depth.
- Fun facts exclude bot edits and trivial changes.

## Failure Mode
If too few viable topics exist, return the best available set and flag the shortfall. Never invent topics.
