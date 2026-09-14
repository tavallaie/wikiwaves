"""TTS Engine — thin wrapper around Kyutai PocketTTS."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import soundfile as sf
from loguru import logger

# Community configs whose name matches this pattern take romanised phonemes,
# not Persian script (e.g. hf://mehdi-hf/pocket-tts-farsi-v2/model.yaml).
_FARSI_V2_CONFIG_RE = re.compile(r"farsi[-_]v2", re.IGNORECASE)

# Training capped voice prompts at 5.0 s; longer prompts make the model continue
# the prompt's own speech instead of the requested text (model-card finding).
_FARSI_V2_MAX_VOICE_SEC = 5.0

# Near-silence after a full generation usually means the model never conditioned
# on the text (or ran away). Surface it instead of writing a quiet WAV.
_MIN_USEFUL_RMS = 0.01


def _needs_farsi_g2p(config: str | None) -> bool:
    """True when `config` names a PocketTTS Farsi v2 checkpoint."""
    return bool(config) and bool(_FARSI_V2_CONFIG_RE.search(config))


def _is_catalog_voice_name(voice: str) -> bool:
    """True for PocketTTS built-in names / Supertonic M1..F5 — not a wav path."""
    if voice.startswith(("http://", "https://", "hf://")):
        return False
    if "/" in voice or "\\" in voice or voice.endswith((".wav", ".flac", ".mp3", ".ogg")):
        return False
    return True


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

        self._farsi_v2 = _needs_farsi_g2p(config)
        # Farsi v2 model-card defaults. The runner's Supertonic-oriented
        # defaults (steps=5, eos=-4, frames_after_eos=None) produce long
        # near-silent runaways on this checkpoint.
        if self._farsi_v2:
            if sampler_decode_steps == 5:
                sampler_decode_steps = 1
            if eos_threshold == -4.0:
                eos_threshold = -2.0
            if frames_after_eos is None:
                frames_after_eos = 0
            if temp is None:
                temp = 0.3

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
        self._community_config = bool(config)

        self._g2p = None
        if self._farsi_v2:
            from wikiwaves.tts.farsi_g2p import FarsiG2P

            try:
                self._g2p = FarsiG2P()
            except ImportError as exc:
                raise RuntimeError(
                    "The 'transformers' package is required for PocketTTS Farsi v2. "
                    "Install it:\n"
                    "  uv sync --extra pockettts\n"
                    "  or: pip install transformers"
                ) from exc

    # ------------------------------------------------------------------ #
    # Voice styles
    # ------------------------------------------------------------------ #

    def get_voice_style(self, voice_name: str) -> object:
        """Load a preset voice, a local audio path, an hf:// URI, or a URL."""
        if self._community_config and _is_catalog_voice_name(voice_name):
            raise ValueError(
                f"Community PocketTTS configs need a voice wav path, hf:// URI, or URL; "
                f"got catalog name {voice_name!r}. Pass --voice path/to/prompt.wav "
                f"(≤{_FARSI_V2_MAX_VOICE_SEC:.0f}s for Farsi v2)."
            )
        if self._farsi_v2:
            return self._get_voice_style_truncated(voice_name)
        return self._tts.get_state_for_audio_prompt(voice_name)

    def get_voice_style_from_path(self, path: str | Path) -> object:
        """Load a custom voice from a local audio path, hf:// URI, or URL."""
        return self.get_voice_style(str(path))

    def _get_voice_style_truncated(self, voice_name: str) -> object:
        """Load a voice prompt, truncating local wavs to 5 seconds for Farsi v2."""
        path = Path(voice_name)
        if not path.is_file():
            return self._tts.get_state_for_audio_prompt(voice_name)

        audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
        # soundfile returns (T, C); model wants [channels, samples]
        audio = audio.T
        max_samples = int(_FARSI_V2_MAX_VOICE_SEC * sr)
        if audio.shape[-1] > max_samples:
            logger.warning(
                f"Voice prompt {path.name} is {audio.shape[-1] / sr:.1f}s; "
                f"truncating to {_FARSI_V2_MAX_VOICE_SEC:.0f}s for Farsi v2"
            )
            audio = audio[..., :max_samples]
        import torch

        return self._tts.get_state_for_audio_prompt(torch.from_numpy(audio))

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

        if self._g2p is not None:
            wav = self._synthesize_phonemized(text.strip(), style, silence_duration)
            self._warn_if_near_silent(wav, text)
            return wav

        voice_label = voice if isinstance(voice, str) else "(style)"
        logger.debug(f"Synthesizing {len(text)} chars with voice {voice_label}…")

        generate_kwargs: dict = {}
        if self.frames_after_eos is not None:
            generate_kwargs["frames_after_eos"] = self.frames_after_eos
        wav = self._tts.generate_audio(style, text.strip(), **generate_kwargs)
        return self._to_numpy(wav)

    def _synthesize_phonemized(
        self,
        text: str,
        style: object,
        silence_duration: float,
    ) -> np.ndarray:
        """Phonemize `text` sentence by sentence and synthesize each part.

        G2P discards punctuation, so sentences are split before
        phonemization and their audio is joined with a short silence gap.
        """
        from wikiwaves.tts.farsi_g2p import split_persian_sentences

        wavs: list[np.ndarray] = []
        for sentence in split_persian_sentences(text):
            phonemes = self._g2p.phonemize(sentence)
            if not phonemes.strip():
                logger.warning(f"Phonemization emptied sentence: {sentence!r}")
                continue

            logger.info(f"Farsi v2 phonemes: {phonemes}")
            arr = self._generate_phonemes_with_retry(style, phonemes)
            wavs.append(arr)

        if not wavs:
            return np.zeros((1, 0), dtype=np.float32)
        return self.concatenate(wavs, silence_sec=silence_duration, sample_rate=self.sample_rate)

    def _generate_phonemes_with_retry(self, style: object, phonemes: str) -> np.ndarray:
        """Generate once; retry when the clip hits the runaway length cap."""
        generate_kwargs: dict = {}
        if self.frames_after_eos is not None:
            generate_kwargs["frames_after_eos"] = self.frames_after_eos

        cap_sec = self._runaway_cap_sec(phonemes)
        wav = self._to_numpy(self._tts.generate_audio(style, phonemes, **generate_kwargs))
        dur = wav.shape[1] / self.sample_rate
        if dur <= cap_sec:
            return wav

        logger.warning(
            f"Runaway generation ({dur:.1f}s > {cap_sec:.1f}s cap) for {phonemes!r}; retrying"
        )
        wav2 = self._to_numpy(self._tts.generate_audio(style, phonemes, **generate_kwargs))
        dur2 = wav2.shape[1] / self.sample_rate
        # Keep the shorter of the two — runaways pad to the length cap.
        if dur2 < dur:
            return wav2
        return wav

    def _runaway_cap_sec(self, phonemes: str) -> float:
        """Seconds above which a clip is treated as a non-terminating runaway.

        Matches the model card: ``tokens / 3.0 + 2.0``.
        """
        try:
            n_tokens = len(self._tts.flow_lm.conditioner.tokenizer.sp.encode(phonemes))
        except Exception:
            n_tokens = max(1, len(phonemes.split()))
        return n_tokens / 3.0 + 2.0

    def _warn_if_near_silent(self, wav: np.ndarray, text: str) -> None:
        if wav.size == 0:
            return
        rms = float(np.sqrt(np.mean(wav.astype(np.float64) ** 2)))
        if rms < _MIN_USEFUL_RMS:
            logger.error(
                f"Generated audio is near-silent (rms={rms:.5f}) for {len(text)} chars. "
                f"Check --voice is a ≤{_FARSI_V2_MAX_VOICE_SEC:.0f}s wav and that G2P ran "
                f"(look for 'Farsi v2 phonemes:' in the log)."
            )

    @staticmethod
    def _to_numpy(wav: object) -> np.ndarray:
        """Convert a torch-like tensor (or array-like) to (1, T) float32."""
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
