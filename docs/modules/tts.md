# TTS Engine — Text-to-Speech

## Purpose
Convert each script chunk into an audio segment. Hide provider-specific details so voices can be swapped without touching the rest of the pipeline.

## Input
- Structured script chunks (text + speaker identifier)
- Voice map (speaker to voice assignment)

## Output
- One audio file per chunk
- Audio manifest listing all generated files in order

## Boundaries
- One chunk = one file. This keeps retry, pause control, and parallel generation simple.
- Validates voice availability before generation begins.
- Supports at least two distinguishable voices.

## Failure Mode
A single chunk failure is retried or logged; the rest of the episode continues. Missing chunks are noted in the manifest for the assembler to handle.
