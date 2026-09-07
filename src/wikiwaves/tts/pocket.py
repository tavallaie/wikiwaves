"""TTS Engine — thin wrapper around Kyutai PocketTTS."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from loguru import logger


class PocketTTSEngine:
    """Lightweight TTS engine using Kyutai PocketTTS."""

    def __init__(
        self,
        language: str | None = "english",
        config: str | None = None,
        sampler_decode_steps: int = 1,
        temp: float | None = None,
        quantize: bool = False,
        eos_threshold: float = -4.0,
        frames_after_eos: int | None = None,
    ):
        try:
            from pocket_tts import TTSModel
        except ImportError as exc:
            raise RuntimeError(
                "The 'pocket-tts' package is required. Install it:\n"
                "  uv pip install pocket-tts\n"
                "  or: pip install pocket-tts"
            ) from exc

        load_kwargs: dict = {
            "sampler_decode_steps": sampler_decode_steps,
            "temp": temp,
            "quantize": quantize,
            "eos_threshold": eos_threshold,
        }
        # language and config are mutually exclusive in TTSModel.load_model.
        if config:
            load_kwargs["config"] = config
        else:
            load_kwargs["language"] = language or "english"

        self._tts = TTSModel.load_model(**load_kwargs)
        self.frames_after_eos = frames_after_eos

    # ------------------------------------------------------------------ #
    # Voice styles
    # ------------------------------------------------------------------ #

    def get_voice_style(self, voice_name: str) -> object:
        """Load a preset voice, a local audio path, an hf:// URI, or a URL."""
        return self._tts.get_state_for_audio_prompt(voice_name)

    def get_voice_style_from_path(self, path: str | Path) -> object:
        """Load a custom voice from a local audio path, hf:// URI, or URL."""
        return self._tts.get_state_for_audio_prompt(str(path))

    def list_voices(self) -> list[str]:
        """Return available preset voice names."""
        return [
            "alba",
            "anna",
            "azelma",
            "bill_boerst",
            "caro_davy",
            "charles",
            "cosette",
            "eponine",
            "eve",
            "fantine",
            "george",
            "jane",
            "jean",
            "javert",
            "marius",
            "mary",
            "michael",
            "paul",
            "peter_yearsley",
            "stuart_bell",
            "vera",
            "giovanni",
            "lola",
            "juergen",
            "rafael",
            "estelle",
        ]

    # ------------------------------------------------------------------ #
    # Synthesis
    # ------------------------------------------------------------------ #

    def synthesize(
        self,
        text: str,
        voice: str | object = "alba",
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

        voice_label = voice if isinstance(voice, str) else "(style)"
        logger.debug(f"Synthesizing {len(text)} chars with voice {voice_label}…")

        generate_kwargs: dict = {}
        if self.frames_after_eos is not None:
            generate_kwargs["frames_after_eos"] = self.frames_after_eos
        wav = self._tts.generate_audio(style, text.strip(), **generate_kwargs)
        if hasattr(wav, "detach"):
            wav = wav.detach()
        if hasattr(wav, "cpu"):
            wav = wav.cpu()
        if hasattr(wav, "numpy"):
            wav = wav.numpy()
        arr = np.asarray(wav, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        return arr

    def save_audio(self, wav: np.ndarray, path: str | Path) -> None:
        """Save a waveform to a WAV file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(path), wav.T, self.sample_rate)
        logger.info(f"Saved audio → {path}")

    # ------------------------------------------------------------------ #
    # Utilities
    # ------------------------------------------------------------------ #

    @property
    def sample_rate(self) -> int:
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
