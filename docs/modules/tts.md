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

## Backends

Two backends are available:

- **supertonic** (default) — ONNX Supertonic. Installed with the project.
- **pockettts** — Kyutai PocketTTS and community checkpoints. Optional extra; pulls PyTorch.

Install PocketTTS:

```bash
uv sync --extra pockettts
```

This installs a fork of `pocket-tts` (`git+https://github.com/mallahyari/pocket-tts@main`) and
`transformers`. The fork is required for community configs whose `model.yaml` sets fields the
PyPI release rejects. `transformers` is used by profiles that need grapheme-to-phoneme (Farsi v2).

### PocketTTS profiles

PocketTTS variants share one engine. Behaviour is selected by a **profile**:

| Profile | When | Text | Voice |
|---------|------|------|-------|
| `official` | `--language` (english, french_24l, …) or no config | as-is | catalog (`alba`, …) or wav |
| `farsi-v2` | config matches `farsi-v2` / `farsi_v2` | Persian → phonemes (G2P) | wav ≤5s required |
| `community` | any other `--config` | as-is | wav required |

Resolution order: `--profile` override → match `--config` → match `--language` → `official`.

```bash
# Official English
uv run python -m wikiwaves.tts.runner 2026-05-11 --backend pockettts --voice alba

# Farsi v2 (auto profile from config)
uv run python -m wikiwaves.tts.runner output/pockettts-farsi-script.txt \
  --backend pockettts \
  --config hf://mehdi-hf/pocket-tts-farsi-v2/model.yaml \
  --voice output/voice-zahra-5s.wav

# Force a profile when auto-detect is wrong
uv run python -m wikiwaves.tts.runner script.txt \
  --backend pockettts \
  --config hf://someone/custom/model.yaml \
  --profile community \
  --voice prompt.wav
```

Default `--voice` is `M1` (Supertonic). When `--backend pockettts` and the voice is still `M1` with no `--config`, the runner uses `alba`.

Add a new community variant by registering a `PocketProfile` in
`src/wikiwaves/tts/pocket_profiles.py` (and a text-prep module if needed).

The first argument can be a date folder under `output/`, an existing directory, or a `.txt` file.

### PocketTTS Farsi v2

`pocket-tts-farsi-v2` takes romanised phonemes, not Persian script. The `farsi-v2` profile
phonemizes each sentence with `mehdi-hf/Homo-GE2PE-Persian-HF` before synthesis — pass plain
Persian text to the runner as usual.

Known limitations, from the model card:

- The voice prompt must be 5 seconds or less. A longer prompt makes the model continue the
  prompt's own speech instead of the requested text.
- Occasionally a generation "runs away" and repeats itself instead of stopping. The profile
  retries once when output exceeds the token-based length cap.
