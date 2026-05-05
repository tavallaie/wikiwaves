# Enricher — Content Expansion

## Purpose
For each selected topic, decide which additional articles would add the most valuable context, then fetch them. This is the "research assistant" layer.

## Input
- A base article for each selected topic
- A list of internal links available within each base article

## Output
- A bundle per topic: base article + validated related articles
- Reasoning for why each related article was chosen

## Boundaries
- Suggestions must come from actual links in the base article; no invented titles.
- Validates every suggestion before fetching.
- Skips redirects, stubs, or unreachable pages gracefully.
- Adds depth, not breadth: 3–5 related articles per topic is enough.

## Failure Mode
If enrichment fails for a topic, emit the base article alone and continue. A thin segment is better than a crashed pipeline.
