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
- **pockettts** — Kyutai PocketTTS. Optional extra; pulls PyTorch.

Install PocketTTS:

```bash
uv sync --extra pockettts
```

This installs a fork of `pocket-tts` (`git+https://github.com/mallahyari/pocket-tts@main`) and
`transformers`. The fork is required for community configs whose `model.yaml` sets fields the
PyPI release rejects, such as `pocket-tts-farsi-v2` below. `transformers` powers the Farsi
grapheme-to-phoneme step.

Select the backend on the runner:

```bash
uv run python -m wikiwaves.tts.runner 2026-05-11 --backend pockettts --voice alba
```

Default `--voice` is `M1` (Supertonic). When `--backend pockettts` and the voice is still `M1`, the runner uses `alba`.

Community models use `--config` instead of a built-in language. Named catalog voices such as `alba` do not apply. Pass a wav path, an `hf://` URI, or a URL:

```bash
uv run python -m wikiwaves.tts.runner output/pockettts-farsi-script.txt \
  --backend pockettts \
  --config hf://mehdi-hf/pocket-tts-farsi-v2/model.yaml \
  --voice output/voice-zahra-5s.wav \
  --steps 1 --temp 0.3 --eos-threshold -2 --frames-after-eos 0
```

The first argument can be a date folder under `output/`, an existing directory, or a `.txt` file.

### PocketTTS Farsi v2

`pocket-tts-farsi-v2` takes romanised phonemes, not Persian script. `PocketTTSEngine`
detects a `farsi-v2` config automatically and phonemizes each sentence with
`mehdi-hf/Homo-GE2PE-Persian-HF` (via `wikiwaves.tts.farsi_g2p`) before synthesis — pass plain
Persian text to the runner as usual.

Known limitations, from the model card:

- The voice prompt must be 5 seconds or less. A longer prompt makes the model continue the
  prompt's own speech instead of the requested text.
- Occasionally a generation "runs away" and repeats itself instead of stopping. Re-run the
  command; a second attempt usually terminates correctly. (The model card's fix — capping
  generation length from the token count and auto-retrying — needs internals `pocket_tts`
  does not expose publicly, so it is not automated here.)

