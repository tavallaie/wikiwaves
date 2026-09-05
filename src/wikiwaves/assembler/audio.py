"""Audio assembly — concatenate segment WAVs into a single episode."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf
from loguru import logger


def concatenate_wavs(
    wavs: list[np.ndarray],
    sample_rate: int,
    silence_sec: float = 0.8,
) -> np.ndarray:
    """Concatenate waveforms with silence gaps.

    Args:
        wavs: List of mono arrays shaped (1, T) or (T,).
        sample_rate: Audio sample rate.
        silence_sec: Seconds of silence between segments.

    Returns:
        Concatenated array shaped (1, total_samples).
    """
    if not wavs:
        return np.zeros((1, 0), dtype=np.float32)

    silence = np.zeros((1, int(silence_sec * sample_rate)), dtype=np.float32)
    parts: list[np.ndarray] = []

    for i, w in enumerate(wavs):
        w = np.atleast_2d(w)
        if w.shape[0] != 1:
            w = w.reshape(1, -1)
        parts.append(w)
        if i < len(wavs) - 1:
            parts.append(silence)

    return np.concatenate(parts, axis=1)


def read_segment_wavs(audio_dir: Path) -> list[tuple[Path, np.ndarray, int]]:
    """Read all WAV files in a directory, sorted.

    Returns list of (path, wav_array, sample_rate).
    """
    files = sorted(audio_dir.glob("*.wav"))
    if not files:
        raise FileNotFoundError(f"No WAV files found in {audio_dir}")

    results: list[tuple[Path, np.ndarray, int]] = []
    for f in files:
        if f.name == "episode.wav":
            continue  # skip previously built episode
        wav, sr = sf.read(str(f))
        wav = np.atleast_2d(wav).astype(np.float32)
        if wav.shape[0] != 1:
            wav = wav.reshape(1, -1)
        results.append((f, wav, sr))

    return results


def build_episode(
    audio_dir: Path,
    output_name: str = "episode.wav",
    silence_sec: float = 0.8,
    namer: Callable[[Path], str] | None = None,
) -> Path:
    """Concatenate all segment WAVs into a single episode file.

    Returns path to the generated episode WAV.
    """
    segments = read_segment_wavs(audio_dir)
    if not segments:
        raise FileNotFoundError(f"No segment WAVs to assemble in {audio_dir}")

    sample_rate = segments[0][2]
    wavs = [wav for _path, wav, _sr in segments]

    episode = concatenate_wavs(wavs, sample_rate, silence_sec=silence_sec)
    output_path = audio_dir / output_name
    sf.write(str(output_path), episode.T, sample_rate)

    duration = episode.shape[1] / sample_rate
    logger.info(
        f"Episode assembled → {output_path} "
        f"({duration / 60:.1f} min, {len(segments)} segments, {sample_rate} Hz)"
    )
    return output_path
