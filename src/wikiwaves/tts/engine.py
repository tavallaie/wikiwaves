"""TTS Engine — thin wrapper around the official supertonic package."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
from loguru import logger

try:
    from supertonic import TTS as _SupertonicTTS
except ImportError as exc:
    raise RuntimeError(
        "The 'supertonic' package is required. Install it:\n"
        "  uv pip install supertonic\n"
        "  or: pip install supertonic"
    ) from exc


class TTSEngine:
    """Lightweight ONNX TTS engine using the official supertonic package."""

    def __init__(
        self,
        auto_download: bool = True,
        intra_op_num_threads: int | None = None,
        inter_op_num_threads: int | None = None,
        total_steps: int = 5,
        speed: float = 1.0,
    ):
        """
        Args:
            auto_download: Download models on first run (~305 MB).
            intra_op_num_threads: ONNX intra-op threads (None = auto).
            inter_op_num_threads: ONNX inter-op threads (None = auto).
            total_steps: Denoising steps (2=fast, 5=balanced, 10=quality).
            speed: Speech speed (0.7=slow, 1.0=normal, 1.3=fast, 2.0=max).
        """
        kwargs: dict = {"auto_download": auto_download}
        if intra_op_num_threads is not None:
            kwargs["intra_op_num_threads"] = intra_op_num_threads
        if inter_op_num_threads is not None:
            kwargs["inter_op_num_threads"] = inter_op_num_threads

        self._tts = _SupertonicTTS(**kwargs)
        self.total_steps = total_steps
        self.speed = speed

    # ------------------------------------------------------------------ #
    # Voice styles
    # ------------------------------------------------------------------ #

    def get_voice_style(self, voice_name: str) -> object:
        """Load a preset voice style (M1..M5, F1..F5)."""
        return self._tts.get_voice_style(voice_name)

    def get_voice_style_from_path(self, path: str | Path) -> object:
        """Load a custom voice style from a JSON file."""
        return self._tts.get_voice_style_from_path(Path(path))

    def list_voices(self) -> list[str]:
        """Return available preset voice names."""
        return list(self._tts.voice_style_names)

    # ------------------------------------------------------------------ #
    # Synthesis
    # ------------------------------------------------------------------ #

    def synthesize(
        self,
        text: str,
        voice: str | object = "M1",
        lang: str = "en",
        total_steps: int | None = None,
        speed: float | None = None,
        max_chunk_length: int = 300,
        silence_duration: float = 0.3,
    ) -> np.ndarray:
        """Synthesize text to a mono audio array.

        Returns:
            np.ndarray of shape (1, num_samples), dtype float32
        """
        if isinstance(voice, str):
            style = self.get_voice_style(voice)
        else:
            style = voice

        if not text or not text.strip():
            return np.zeros((1, 0), dtype=np.float32)

        steps = total_steps if total_steps is not None else self.total_steps
        spd = speed if speed is not None else self.speed
        voice_label = voice if isinstance(voice, str) else "(style)"

        logger.debug(
            f"Synthesizing {len(text)} chars with voice {voice_label} "
            f"(steps={steps}, speed={spd})…"
        )

        wav, _duration = self._tts.synthesize(
            text.strip(),
            voice_style=style,
            lang=lang,
            total_steps=steps,
            speed=spd,
            max_chunk_length=max_chunk_length,
            silence_duration=silence_duration,
        )
        return wav  # shape (1, T)

    def save_audio(self, wav: np.ndarray, path: str | Path) -> None:
        """Save a waveform to a WAV file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._tts.save_audio(wav, str(path))
        logger.info(f"Saved audio → {path}")

    # ------------------------------------------------------------------ #
    # Utilities
    # ------------------------------------------------------------------ #

    @property
    def sample_rate(self) -> int:
        # Supertonic v3 outputs at 24 kHz; query the model if exposed,
        # otherwise fall back to the known default.
        return getattr(self._tts, "sample_rate", 24000)

    @staticmethod
    def concatenate(
        wavs: list[np.ndarray],
        silence_sec: float = 0.5,
        sample_rate: int = 24000,
    ) -> np.ndarray:
        """Concatenate multiple waveforms with silence gaps."""
        if not wavs:
            return np.zeros((1, 0), dtype=np.float32)

        silence = np.zeros((1, int(silence_sec * sample_rate)), dtype=np.float32)
        parts = []
        for i, w in enumerate(wavs):
            parts.append(w)
            if i < len(wavs) - 1:
                parts.append(silence)
        return np.concatenate(parts, axis=1)


def create_engine(
    auto_download: bool = True,
    total_steps: int = 5,
    speed: float = 1.0,
    backend: str = "supertonic",
) -> TTSEngine:
    """Factory — creates and warms up the engine."""
    if backend in ("supertonic", "super"):
        return TTSEngine(auto_download=auto_download, total_steps=total_steps, speed=speed)
    if backend in ("pockettts", "pocket"):
        from wikiwaves.tts.pocket import PocketTTSEngine

        return PocketTTSEngine(sampler_decode_steps=total_steps)
    raise ValueError(f"Unknown TTS backend: {backend!r}")
