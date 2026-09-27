# Plan — Add PocketTTS backend

## Objective

Add Kyutai PocketTTS as a second TTS backend, selectable from the TTS runner, without changing the default Supertonic path.

## Context

- Repository: wikiwaves
- Current branch: main
- Ticket: feat-pockettts (no project ticket key)
- PocketTTS Python API (verified): `pocket_tts.TTSModel.load_model()`, `get_state_for_audio_prompt()`, `generate_audio()`, `sample_rate`
- Package name: `pocket-tts` (import: `pocket_tts`)
- User asked for PocketTTS support, so adding that optional package is approved.

## Impact

| File | Change |
|------|--------|
| `src/wikiwaves/tts/pocket.py` | New. `PocketTTSEngine` with the same methods the runner already calls on `TTSEngine`. |
| `src/wikiwaves/tts/engine.py` | Extend `create_engine()` with a `backend` argument. Lazy-import PocketTTS. |
| `src/wikiwaves/tts/__init__.py` | Export `PocketTTSEngine` and the extended factory. |
| `src/wikiwaves/tts/runner.py` | Add `--backend`. Pick default voice per backend. Build the engine through `create_engine()`. |
| `tests/test_tts.py` | New. Mock `pocket_tts`. Do not load the real model. |
| `pyproject.toml` | Optional extra `pockettts = ["pocket-tts"]`. |
| `docs/modules/tts.md` | Document the two backends and how to select PocketTTS. |

Callers of `TTSEngine` today: `runner.py` only. Assembler reads WAV files and does not construct an engine.

Do not delete files.

## Contract impact

- No API JSON keys.
- No background jobs.
- No database.
- New optional dependency: `pocket-tts` (pulls PyTorch). Keep it off the default install.
- CLI: new `--backend {supertonic,pockettts}`. Default stays `supertonic`.
- CLI: `--voice` default stays `M1` for Supertonic. When `--backend pockettts` and voice is still `M1`, use `alba`.
- Waveform contract stays `np.ndarray` shape `(1, T)`, float32. PocketTTS 1D torch tensors convert to that shape.

## Size

| File | Expected lines |
|------|----------------|
| `pocket.py` | ~130 |
| `engine.py` | ~25 |
| `__init__.py` | ~8 |
| `runner.py` | ~40 |
| `test_tts.py` | ~180 |
| `pyproject.toml` | ~4 |
| `docs/modules/tts.md` | ~25 |

## Pattern

Copy style from:

1. `src/wikiwaves/tts/engine.py` — thin wrapper, loguru, `RuntimeError` on missing package, `synthesize` returns `(1, T)`.
2. `src/wikiwaves/tts/runner.py` — argparse, `run()`, voice mapping.
3. `tests/test_llm.py` — unittest + `unittest.mock`.

Do not add a Protocol or ABC. Duck-type the same method names.

## Tests

Write these failing tests first. Mock `pocket_tts.TTSModel`. Do not download weights.

1. Missing `pocket_tts` raises `RuntimeError`.
2. `synthesize` converts a 1D torch-like tensor to `np.ndarray` shape `(1, T)` dtype float32.
3. Empty text returns an empty `(1, 0)` array and does not call `generate_audio`.
4. `get_voice_style("alba")` calls `get_state_for_audio_prompt("alba")`.
5. `get_voice_style` with a local path passes that path through.
6. `list_voices` returns the documented catalog names including `alba`.
7. `create_engine(backend="pockettts")` returns `PocketTTSEngine`.
8. `create_engine(backend="supertonic")` still returns `TTSEngine` (mock supertonic if import is heavy).
9. `create_engine(backend="nope")` raises `ValueError`.
10. `run(..., backend="pockettts")` uses PocketTTS and writes a WAV (mock engine).

## Steps

1. Write `tests/test_tts.py` with the cases above. Confirm they fail.
2. Add `PocketTTSEngine` in `pocket.py`.
   - Lazy import `TTSModel`.
   - `load_model(language=..., sampler_decode_steps=..., temp=..., quantize=...)`.
   - `get_voice_style(name)` → `get_state_for_audio_prompt(name)`.
   - Named voices from the PocketTTS catalog: alba, anna, azelma, bill_boerst, caro_davy, charles, cosette, eponine, eve, fantine, george, jane, jean, javert, marius, mary, michael, paul, peter_yearsley, stuart_bell, vera, giovanni, lola, juergen, rafael, estelle.
   - Also accept a local audio path, `hf://...`, or a URL (the library already does).
   - `synthesize` ignores Supertonic-only kwargs (`speed`, `max_chunk_length`, `silence_duration`) so the runner does not branch.
   - `save_audio` uses `soundfile` (already a dependency).
   - `concatenate` copies the numpy concat already on `TTSEngine` (do not import Supertonic from pocket.py).
3. Extend `create_engine(backend="supertonic"|"pockettts")`.
4. Wire `runner.run()` and argparse `--backend`.
5. Add optional extra in `pyproject.toml`. Do not add `pocket-tts` to the default dependencies.
6. Update `docs/modules/tts.md`.
7. Run verify.

## Risk

| Risk | How we find it |
|------|----------------|
| PyTorch in the default env | Optional extra only. Tests mock the package. |
| Voice `M1` sent to PocketTTS | Runner remaps default `M1` to `alba` only when backend is `pockettts`. |
| Shape mismatch with assembler | Convert to `(1, T)` in `synthesize`. Tests check shape. |
| Importing `pocket.py` pulls torch | Import `TTSModel` inside `__init__`, same pattern as `engine.py`. |
| `create_engine` in `engine.py` still imports supertonic | Existing behaviour. Default backend stays Supertonic. |

## Verification

AGENTS.md does not list a verify command. README does:

```bash
uv run python -m unittest discover -s tests -v
```

## Increment 2 — Switch Farsi PocketTTS to v2 (G2P phonemization)

### Objective

Replace the Farsi PocketTTS example/config with `mehdi-hf/pocket-tts-farsi-v2`, which needs romanised-phoneme input instead of raw Persian script.

### Context

- v1 (`mehdi-hf/pocket-tts-farsi`) took Persian script directly. v2 does not: it silently emits near-silence on raw Persian, because its vocabulary has no entry for Persian codepoints.
- v2 needs a separate grapheme-to-phoneme model, `mehdi-hf/Homo-GE2PE-Persian-HF` (a `transformers` T5 model), plus text normalization shipped as `normalize_fa.py` on the v2 model card.
- v2's `model.yaml` sets config keys (`capitalize_first_letter`, `append_terminal_punctuation`, `pad_with_spaces_for_short_inputs`) that the PyPI `pocket-tts` release rejects. The model card names a fork that supports them: `git+https://github.com/mallahyari/pocket-tts@main`.
- User approved: switch the `pockettts` extra to the fork, add `transformers` to that same extra, auto-detect v2 configs inside `PocketTTSEngine` (no new CLI flag), and replace the v1 references in docs/examples/tests.

### Impact

| File | Change |
|------|--------|
| `pyproject.toml` | `pockettts` extra now installs the pocket-tts fork (git) and `transformers`. |
| `src/wikiwaves/tts/farsi_normalize.py` | New. Trimmed, vendored copy of the Persian text-normalization functions from the v2 model card (no CLI, no `typer` dependency). |
| `src/wikiwaves/tts/farsi_g2p.py` | New. `FarsiG2P` (lazy `transformers` load) and `split_persian_sentences()`. |
| `src/wikiwaves/tts/pocket.py` | Auto-detect a v2 config in `__init__`; phonemize per sentence and concatenate in `synthesize()`. |
| `src/wikiwaves/tts/runner.py` | Update the module docstring example to the v2 config path. |
| `tests/test_tts.py` | Mock `transformers`; add tests for phonemization detection, sentence splitting, and the concatenated output. |
| `docs/modules/tts.md` | Replace the v1 config example and note the new install footprint and phoneme requirement. |

### Contract impact

- No API JSON keys, no background jobs, no database.
- New dependency inside the existing `pockettts` extra: `transformers`. Still off the default install.
- `pockettts` extra now installs `pocket-tts` from a git fork instead of PyPI.
- Waveform contract unchanged: `(1, T)` float32.

### Scope cut (explicit)

The model card also documents a "runaway generation" retry (regenerate when output exceeds `tokens / 3.0 + 2.0` seconds). `pocket_tts.TTSModel`'s public API does not expose the token count needed to compute that cap. Skipping the retry for this increment; documented as a known limitation instead of implemented.

### Pattern

Copy style from `src/wikiwaves/tts/pocket.py` (lazy import, `RuntimeError` on missing package) and `src/wikiwaves/tts/engine.py`.

### Tests

1. `PocketTTSEngine` with a v2-style config auto-enables phonemization.
2. `PocketTTSEngine` with a non-v2 config (or no config) does not phonemize.
3. `synthesize()` on multi-sentence Persian text calls `generate_audio` once per sentence and concatenates.
4. Missing `transformers` raises `RuntimeError` mentioning `transformers`, only when a v2 config is used.
5. `split_persian_sentences()` splits on `.`, `!`, `؟`.

### Steps

1. Vendor `farsi_normalize.py`.
2. Add `farsi_g2p.py`.
3. Wire `pocket.py` auto-detection and phonemized synthesis path.
4. Update `pyproject.toml`, docs, runner docstring.
5. Write/update tests.
6. Verify.

### Risk

| Risk | How we find it |
|------|-----------------|
| v2 detection regex misses a config string variant | Match on `farsi-v2` case-insensitively; test both `hf://` and bare filename forms. |
| `transformers` heavy import slows non-Farsi PocketTTS use | Only imported when a v2 config is detected. |

### Verification

```bash
uv run python -m unittest discover -s tests -v
```
