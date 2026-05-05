# Utils — Shared Utilities

## Purpose
Hold cross-cutting helpers used by multiple modules. Keep modules DRY without creating hidden coupling.

## Input
- Varied; utility functions are stateless and generic.

## Output
- Clean text, retry wrappers, logging setup, folder management, etc.

## Boundaries
- Utilities are pure and side-effect-free where possible.
- No business logic belongs here.
- Any module may import utils, but utils must never import module-specific code.

## Failure Mode
Utilities raise clear exceptions and never swallow errors silently.
