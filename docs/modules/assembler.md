# Assembler — Audio Stitching

## Purpose
Combine all individual chunk audio files into one seamless podcast episode. Add structural silence and optional music.

## Input
- Audio files in script order
- Pause metadata per chunk
- Optional intro/outro music files

## Output
- A single continuous audio file for the episode
- Metadata tags (title, date, description)

## Boundaries
- Inserts exact silence durations between chunks.
- Normalizes overall loudness to a podcast standard.
- Music, if provided, fades in/out around speech; it never overpowers.
- Does not edit chunk audio content, only concatenates and spaces.

## Failure Mode
If a chunk file is missing, insert a brief silence and continue. The episode should always ship, even if slightly imperfect.
