# Scripter — Dialogue Generation

## Purpose
Transform aggregated factual content into a natural, engaging two-person conversation. This is the creative heart of the pipeline.

## Input
- Enriched topic bundles (base + related articles)
- Fun facts
- Speaker personas and style constraints

## Output
- A complete episode script in a structured chunk format
- Grounding report mapping claims to source articles

## Boundaries
- Every factual claim must be traceable to a source article.
- Lines should be short, conversational, and paced for speech.
- Balances accuracy with natural flow; never reads like an encyclopedia entry.
- Intro, segments, and outro are all produced here.

## Failure Mode
If a segment cannot be grounded, flag it in the grounding report rather than silently inventing facts.
