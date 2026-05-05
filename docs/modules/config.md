# Config — Centralized Settings

## Purpose
Single source of truth for all configurable values. Nothing else in the system hardcodes a constant.

## Input
- Environment variables
- Optional local override files

## Output
- Structured configuration object consumed by all modules

## Boundaries
- Covers every external boundary: API endpoints, rate limits, voice IDs, file paths, episode length, retry policies.
- Validates required values at startup and fails fast if missing.
- Changing one value here changes behavior everywhere without code edits.

## Failure Mode
Missing required config aborts startup immediately with a clear message. No silent defaults for credentials or critical paths.
