# Orchestrator — Pipeline Runner

## Purpose
Wire all modules together in the correct order, manage data handoff between stages, and produce a final report.

## Input
- A target date
- Configuration values
- All module interfaces

## Output
- Final episode artifacts (audio, optional video)
- A human-readable run report
- Organized dated output folder

## Boundaries
- Knows the sequence; delegates all work to specialized modules.
- Cleans up temporary files older than a configurable threshold.
- Runs each stage with timing and logging.
- Never hardcodes module internals.

## Failure Mode
Catches and logs errors per stage. A stage failure does not automatically kill the whole pipeline unless configured to do so. Always produces a report.
