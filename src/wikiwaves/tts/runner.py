"""TTS Runner — flexible podcast audio generation from scripts.

Usage:
    uv run python -m wikiwaves.tts.runner 2026-05-11
    uv run python -m wikiwaves.tts.runner 2026-05-11 --voice M2 --host-voice F1 --steps 5
    uv run python -m wikiwaves.tts.runner 2026-05-11 --backend pockettts --voice alba
    uv run python -m wikiwaves.tts.runner output/pockettts-farsi-script.txt --backend pockettts \
        --config hf://mehdi-hf/pocket-tts-farsi/farsi.yaml \
        --voice output/voice-zahra-5s.wav
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from loguru import logger

from wikiwaves.tts.engine import TTSEngine, create_engine


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #

@dataclass
class ScriptSegment:
    """A single piece of the podcast script."""

    seg_type: str  # intro, topic, transition, outro
    title: str
    text: str
    voice: str | None = None  # per-segment override
    metadata: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Readers
# --------------------------------------------------------------------------- #

def read_scripts_json(day_dir: Path) -> list[ScriptSegment]:
    """Read scripts.json (new format with intro included)."""
    path = day_dir / "scripts.json"
    if not path.exists():
        raise FileNotFoundError(f"No scripts.json found in {day_dir}")

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    segments: list[ScriptSegment] = []
    for item in raw:
        segments.append(
            ScriptSegment(
                seg_type=item.get("type", "topic"),
                title=item.get("title", ""),
                text=item.get("text", ""),
                metadata={k: v for k, v in item.items() if k not in ("type", "title", "text")},
            )
        )
    return segments


def read_script_txt(path: Path) -> ScriptSegment:
    """Parse any .txt file into a ScriptSegment."""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    lines = content.splitlines()
    fname = path.stem.lower()
    has_header = any(
        line.strip().startswith("Duration:") or line.strip().startswith("-" * 3)
        for line in lines
    )

    # A plain text file (no intro/transition name, no header) is spoken as-is.
    if "intro" not in fname and "transition" not in fname and not has_header:
        return ScriptSegment(seg_type="topic", title=path.stem, text=content.strip())

    # Detect type from filename content
    if "intro" in fname:
        seg_type = "intro"
        title = "Episode Intro"
    elif "transition" in fname:
        seg_type = "transition"
        title = lines[0].strip() if lines else "Transition"
    else:
        seg_type = "topic"
        title = lines[0].strip() if lines else path.stem

    # Strip header boilerplate (title line, duration line, dashes)
    body_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("Duration:") or stripped.startswith("-") * 3:
            continue
        if stripped == title:
            continue
        body_lines.append(line)

    text = "\n".join(body_lines).strip()
    return ScriptSegment(seg_type=seg_type, title=title, text=text)


def read_scripts_from_txts(
    day_dir: Path,
    exclude: list[str] | None = None,
) -> list[ScriptSegment]:
    """Read all .txt files in a directory, sorted, excluding patterns.

    Args:
        exclude: glob patterns to skip (default skips topic_*.txt).
    """
    exclude = exclude or ["topic_*.txt"]
    all_txt = sorted(day_dir.glob("*.txt"))

    files: list[Path] = []
    for p in all_txt:
        if any(p.match(pat) for pat in exclude):
            continue
        files.append(p)

    if not files:
        raise FileNotFoundError(f"No .txt script files found in {day_dir}")
    return [read_script_txt(f) for f in files]


def read_scripts(day_dir: Path, prefer: str = "txt") -> list[ScriptSegment]:
    """Auto-detect and read scripts from a date folder.

    Defaults to .txt files. Falls back to scripts.json only if requested
    or if no .txt files are present.
    """
    json_path = day_dir / "scripts.json"
    txt_files = [p for p in day_dir.glob("*.txt") if not p.match("topic_*.txt")]

    if prefer == "txt" and txt_files:
        return read_scripts_from_txts(day_dir)
    if json_path.exists():
        return read_scripts_json(day_dir)
    if txt_files:
        return read_scripts_from_txts(day_dir)

    raise FileNotFoundError(f"No scripts found in {day_dir}")


def resolve_script_input(
    source: str,
    output_dir: str,
    prefer: str = "txt",
) -> tuple[list[ScriptSegment], Path]:
    """Load scripts from a date key, an existing directory, or a .txt file.

    Returns (segments, audio_dir).
    """
    raw = Path(source)
    if raw.is_file() and raw.suffix.lower() == ".txt":
        return [read_script_txt(raw)], raw.parent / "audio"
    if raw.is_dir():
        return read_scripts(raw, prefer=prefer), raw / "audio"
    day_dir = Path(output_dir) / source
    return read_scripts(day_dir, prefer=prefer), day_dir / "audio"


# --------------------------------------------------------------------------- #
# Text normalisation (prevents TTS repeats/artifacts)
# --------------------------------------------------------------------------- #

def normalize_text_for_tts(text: str) -> str:
    """Clean text before sending to TTS to reduce repeats and glitches."""
    # Collapse all whitespace to single spaces
    text = re.sub(r"\s+", " ", text)

    # Remove markdown / formatting characters
    text = re.sub(r"[*_`#~|]", "", text)

    # Replace URLs with "link"
    text = re.sub(
        r"https?://\S+|www\.\S+",
        "link",
        text,
    )

    # Normalise dashes / hyphens
    text = text.replace("–", "-").replace("—", "-").replace("‑", "-")

    # Normalise quotes
    text = (
        text.replace("“", '"')
        .replace("”", '"')
        .replace("‘", "'")
        .replace("’", "'")
    )

    # Remove stray brackets / braces that can confuse the model
    text = re.sub(r"[\[\]{}<>]", "", text)

    # Strip leading / trailing whitespace
    text = text.strip()

    return text


# --------------------------------------------------------------------------- #
# Voice assignment
# --------------------------------------------------------------------------- #

def default_voice_mapper(
    segment: ScriptSegment,
    *,
    narrator_voice: str,
    host_voice: str | None = None,
) -> str:
    if segment.voice:
        return segment.voice
    if segment.seg_type in ("intro", "transition", "outro"):
        return host_voice or narrator_voice
    return narrator_voice


# --------------------------------------------------------------------------- #
# Naming
# --------------------------------------------------------------------------- #

def default_namer(segment: ScriptSegment, index: int) -> str:
    safe = re.sub(r"[^\w]", "_", segment.title[:40], flags=re.UNICODE).strip("_")
    return f"{index:02d}_{segment.seg_type}_{safe}.wav"


# --------------------------------------------------------------------------- #
# Synthesis pipeline
# --------------------------------------------------------------------------- #

def synthesize_segments(
    engine: TTSEngine,
    segments: list[ScriptSegment],
    *,
    narrator_voice: str = "M1",
    host_voice: str | None = None,
    voice_mapper: Callable[[ScriptSegment, str, str | None], str] | None = None,
    lang: str = "en",
    total_steps: int | None = None,
    speed: float | None = None,
    max_chunk_length: int = 300,
    silence_duration: float = 0.3,
) -> list[tuple[ScriptSegment, np.ndarray]]:
    """Synthesize audio for every segment.

    Returns a list of (segment, wav_array) pairs in input order.
    """
    mapper = voice_mapper or default_voice_mapper
    results: list[tuple[ScriptSegment, np.ndarray]] = []

    # Pre-load voice styles
    unique_voices: set[str] = set()
    for seg in segments:
        if not seg.text.strip():
            continue
        unique_voices.add(mapper(seg, narrator_voice=narrator_voice, host_voice=host_voice))

    styles: dict[str, object] = {}
    for v in unique_voices:
        styles[v] = engine.get_voice_style(v)

    for i, seg in enumerate(segments):
        raw_text = seg.text.strip()
        if not raw_text:
            logger.warning(f"Skipping empty segment: {seg.title}")
            continue

        text = normalize_text_for_tts(raw_text)
        voice = mapper(seg, narrator_voice=narrator_voice, host_voice=host_voice)
        logger.info(
            f"[{i + 1}/{len(segments)}] Synthesizing {seg.seg_type}: {seg.title} "
            f"({len(text)} chars) with voice {voice}…"
        )

        try:
            wav = engine.synthesize(
                text,
                voice=styles[voice],
                lang=lang,
                total_steps=total_steps,
                speed=speed,
                max_chunk_length=max_chunk_length,
                silence_duration=silence_duration,
            )
        except Exception as exc:
            logger.error(f"Failed to synthesize '{seg.title}': {exc}")
            continue

        results.append((seg, wav))

    return results


# --------------------------------------------------------------------------- #
# Writers
# --------------------------------------------------------------------------- #

def save_segments(
    results: list[tuple[ScriptSegment, np.ndarray]],
    audio_dir: Path,
    namer: Callable[[ScriptSegment, int], str] | None = None,
    engine: TTSEngine | None = None,
) -> list[Path]:
    """Save each segment as its own WAV file."""
    namer = namer or default_namer
    audio_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    for idx, (seg, wav) in enumerate(results):
        fname = namer(seg, idx)
        path = audio_dir / fname
        if engine:
            engine.save_audio(wav, path)
        else:
            import soundfile as sf
            sf.write(str(path), wav.T, 24000)
        paths.append(path)

    return paths


def concatenate_results(
    results: list[tuple[ScriptSegment, np.ndarray]],
    output_path: Path,
    engine: TTSEngine | None = None,
    silence_sec: float = 0.8,
) -> Path:
    """Concatenate all segment waveforms into one file."""
    if not results:
        raise ValueError("No results to concatenate")

    wavs = [wav for _seg, wav in results]
    sr = engine.sample_rate if engine else 24000
    episode = TTSEngine.concatenate(wavs, silence_sec=silence_sec, sample_rate=sr)

    if engine:
        engine.save_audio(episode, output_path)
    else:
        import soundfile as sf
        sf.write(str(output_path), episode.T, sr)

    duration = episode.shape[1] / sr
    logger.info(
        f"Episode complete → {output_path} "
        f"({duration / 60:.1f} min, {len(results)} segments)"
    )
    return output_path


# --------------------------------------------------------------------------- #
# High-level run()
# --------------------------------------------------------------------------- #

def run(
    date_str: str | None = None,
    output_dir: str = "output",
    voice: str = "M1",
    host_voice: str | None = None,
    total_steps: int = 5,
    speed: float = 1.0,
    silence_between: float = 0.8,
    concat: bool = False,
    prefer: str = "txt",
    namer: Callable[[ScriptSegment, int], str] | None = None,
    voice_mapper: Callable[[ScriptSegment, str, str | None], str] | None = None,
    max_chunk_length: int = 300,
    silence_duration: float = 0.3,
    backend: str = "supertonic",
    config: str | None = None,
    language: str | None = None,
    temp: float | None = None,
    eos_threshold: float | None = None,
    frames_after_eos: int | None = None,
) -> list[Path]:
    """Generate audio for all scripts in a date folder, directory, or .txt file."""
    date_str = date_str or datetime.date.today().isoformat()
    segments, audio_dir = resolve_script_input(date_str, output_dir, prefer=prefer)
    if not segments:
        logger.warning("No scripts found.")
        return []

    # Named catalog voices such as alba are not valid for community configs.
    if backend in ("pockettts", "pocket") and voice == "M1" and not config:
        voice = "alba"

    engine = create_engine(
        backend=backend,
        total_steps=total_steps,
        speed=speed,
        config=config,
        language=language,
        temp=temp,
        eos_threshold=eos_threshold,
        frames_after_eos=frames_after_eos,
    )
    logger.info(
        f"TTS engine ready — default voice: {voice}, steps: {total_steps}, speed: {speed}"
    )

    results = synthesize_segments(
        engine,
        segments,
        narrator_voice=voice,
        host_voice=host_voice,
        voice_mapper=voice_mapper,
        total_steps=total_steps,
        speed=speed,
        max_chunk_length=max_chunk_length,
        silence_duration=silence_duration,
    )

    paths = save_segments(results, audio_dir, namer=namer, engine=engine)

    if concat and results:
        concatenate_results(
            results, audio_dir / "episode.wav", engine=engine, silence_sec=silence_between
        )

    logger.info(f"Done. Generated {len(paths)} segment files in {audio_dir}")
    return paths


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate podcast audio from scripts.")
    parser.add_argument(
        "date",
        nargs="?",
        help="Date folder (YYYY-MM-DD), a script directory, or a .txt file. Defaults to today.",
    )
    parser.add_argument("--output-dir", default="output", help="Output directory.")
    parser.add_argument("--voice", default="M1", help="Default narrator voice.")
    parser.add_argument("--host-voice", default=None, help="Host voice for intro/transitions.")
    parser.add_argument("--steps", type=int, default=5, help="Denoising steps (2=fast, 5=balanced, 10=quality).")
    parser.add_argument("--speed", type=float, default=1.0, help="Speech speed (0.7=slow, 1.0=normal, 1.3=fast, 2.0=max).")
    parser.add_argument("--silence", type=float, default=0.8, help="Silence between segments (sec).")
    parser.add_argument("--concat", action="store_true", help="Also concatenate into episode.wav (deprecated: use assembler instead).")
    parser.add_argument(
        "--prefer", default="txt", choices=["json", "txt"], help="Prefer .txt files or scripts.json."
    )
    parser.add_argument(
        "--backend",
        default="supertonic",
        choices=["supertonic", "pockettts"],
        help="TTS backend (supertonic or pockettts).",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="PocketTTS model config (path, https://, or hf://). Incompatible with --language.",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="PocketTTS built-in language config (english, french_24l, ...).",
    )
    parser.add_argument("--temp", type=float, default=None, help="PocketTTS sampling temperature.")
    parser.add_argument(
        "--eos-threshold",
        type=float,
        default=None,
        help="PocketTTS end-of-sequence threshold.",
    )
    parser.add_argument(
        "--frames-after-eos",
        type=int,
        default=None,
        help="PocketTTS frames to generate after EOS.",
    )
    args = parser.parse_args()

    run(
        date_str=args.date,
        output_dir=args.output_dir,
        voice=args.voice,
        host_voice=args.host_voice,
        total_steps=args.steps,
        speed=args.speed,
        silence_between=args.silence,
        concat=args.concat,
        prefer=args.prefer,
        backend=args.backend,
        config=args.config,
        language=args.language,
        temp=args.temp,
        eos_threshold=args.eos_threshold,
        frames_after_eos=args.frames_after_eos,
    )
