"""TTS Engine — thin wrapper around Kyutai PocketTTS."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from loguru import logger

from wikiwaves.tts.pocket_profiles import PocketProfile, resolve_profile

# Near-silence after a full generation usually means the model never conditioned
# on the text (or ran away). Surface it instead of writing a quiet WAV.
_MIN_USEFUL_RMS = 0.01


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
        profile: str | None = None,
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

        self.profile: PocketProfile = resolve_profile(
            profile=profile, config=config, language=language
        )
        sampler_decode_steps, temp, eos_threshold, frames_after_eos = self._apply_profile_defaults(
            sampler_decode_steps, temp, eos_threshold, frames_after_eos
        )

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

        self._text_prep = None
        if self.profile.text_prep == "farsi_g2p":
            self._text_prep = self._load_farsi_g2p()
        elif self.profile.text_prep != "passthrough":
            raise ValueError(f"Unknown text_prep {self.profile.text_prep!r}")

        logger.info(f"PocketTTS profile: {self.profile.id}")

    def _apply_profile_defaults(
        self,
        sampler_decode_steps: int,
        temp: float | None,
        eos_threshold: float,
        frames_after_eos: int | None,
    ) -> tuple[int, float | None, float, int | None]:
        p = self.profile
        if p.replace_runner_step_default and sampler_decode_steps == 5 and p.default_steps is not None:
            sampler_decode_steps = p.default_steps
        if p.replace_runner_eos_default and eos_threshold == -4.0 and p.default_eos_threshold is not None:
            eos_threshold = p.default_eos_threshold
        if frames_after_eos is None and p.default_frames_after_eos is not None:
            frames_after_eos = p.default_frames_after_eos
        if temp is None and p.default_temp is not None:
            temp = p.default_temp
        return sampler_decode_steps, temp, eos_threshold, frames_after_eos

    @staticmethod
    def _load_farsi_g2p():
        from wikiwaves.tts.farsi_g2p import FarsiG2P

        try:
            return FarsiG2P()
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
        if self.profile.voice_mode == "wav_required" and _is_catalog_voice_name(voice_name):
            limit = self.profile.max_voice_sec
            limit_hint = f" (≤{limit:.0f}s)" if limit else ""
            raise ValueError(
                f"Profile {self.profile.id!r} needs a voice wav path, hf:// URI, or URL; "
                f"got catalog name {voice_name!r}. Pass --voice path/to/prompt.wav{limit_hint}."
            )
        if self.profile.max_voice_sec is not None:
            return self._get_voice_style_truncated(voice_name, self.profile.max_voice_sec)
        return self._tts.get_state_for_audio_prompt(voice_name)

    def get_voice_style_from_path(self, path: str | Path) -> object:
        """Load a custom voice from a local audio path, hf:// URI, or URL."""
        return self.get_voice_style(str(path))

    def _get_voice_style_truncated(self, voice_name: str, max_sec: float) -> object:
        """Load a voice prompt, truncating local wavs to ``max_sec``."""
        path = Path(voice_name)
        if not path.is_file():
            return self._tts.get_state_for_audio_prompt(voice_name)

        audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
        # soundfile returns (T, C); model wants [channels, samples]
        audio = audio.T
        max_samples = int(max_sec * sr)
        if audio.shape[-1] > max_samples:
            logger.warning(
                f"Voice prompt {path.name} is {audio.shape[-1] / sr:.1f}s; "
                f"truncating to {max_sec:.0f}s for profile {self.profile.id}"
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

        if self._text_prep is not None:
            wav = self._synthesize_prepared(text.strip(), style, silence_duration)
            if self.profile.warn_near_silent:
                self._warn_if_near_silent(wav, text)
            return wav

        voice_label = voice if isinstance(voice, str) else "(style)"
        logger.debug(f"Synthesizing {len(text)} chars with voice {voice_label}…")
        return self._generate_once(style, text.strip())

    def _synthesize_prepared(
        self,
        text: str,
        style: object,
        silence_duration: float,
    ) -> np.ndarray:
        """Split, prepare each sentence, synthesize, concatenate."""
        from wikiwaves.tts.farsi_g2p import split_persian_sentences

        wavs: list[np.ndarray] = []
        for sentence in split_persian_sentences(text):
            prepared = self._text_prep.phonemize(sentence)
            if not prepared.strip():
                logger.warning(f"Text prep emptied sentence: {sentence!r}")
                continue

            logger.info(f"PocketTTS [{self.profile.id}] prepared: {prepared}")
            if self.profile.runaway_retry:
                arr = self._generate_with_retry(style, prepared)
            else:
                arr = self._generate_once(style, prepared)
            wavs.append(arr)

        if not wavs:
            return np.zeros((1, 0), dtype=np.float32)
        return self.concatenate(wavs, silence_sec=silence_duration, sample_rate=self.sample_rate)

    def _generate_once(self, style: object, text: str) -> np.ndarray:
        generate_kwargs: dict = {}
        if self.frames_after_eos is not None:
            generate_kwargs["frames_after_eos"] = self.frames_after_eos
        return self._to_numpy(self._tts.generate_audio(style, text, **generate_kwargs))

    def _generate_with_retry(self, style: object, text: str) -> np.ndarray:
        """Generate once; retry when the clip hits the runaway length cap."""
        cap_sec = self._runaway_cap_sec(text)
        wav = self._generate_once(style, text)
        dur = wav.shape[1] / self.sample_rate
        if dur <= cap_sec:
            return wav

        logger.warning(
            f"Runaway generation ({dur:.1f}s > {cap_sec:.1f}s cap) for {text!r}; retrying"
        )
        wav2 = self._generate_once(style, text)
        dur2 = wav2.shape[1] / self.sample_rate
        # Keep the shorter of the two — runaways pad to the length cap.
        if dur2 < dur:
            return wav2
        return wav

    def _runaway_cap_sec(self, text: str) -> float:
        """Seconds above which a clip is treated as a non-terminating runaway.

        Matches the Farsi v2 model card: ``tokens / 3.0 + 2.0``.
        """
        try:
            n_tokens = len(self._tts.flow_lm.conditioner.tokenizer.sp.encode(text))
        except Exception:
            n_tokens = max(1, len(text.split()))
        return n_tokens / 3.0 + 2.0

    def _warn_if_near_silent(self, wav: np.ndarray, text: str) -> None:
        if wav.size == 0:
            return
        rms = float(np.sqrt(np.mean(wav.astype(np.float64) ** 2)))
        if rms < _MIN_USEFUL_RMS:
            limit = self.profile.max_voice_sec
            limit_hint = f"≤{limit:.0f}s " if limit else ""
            logger.error(
                f"Generated audio is near-silent (rms={rms:.5f}) for {len(text)} chars. "
                f"Check --voice is a {limit_hint}wav and that profile {self.profile.id} "
                f"text prep ran (look for 'prepared:' in the log)."
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
