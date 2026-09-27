"""Grapheme-to-phoneme conversion for PocketTTS Farsi v2.

The `mehdi-hf/pocket-tts-farsi-v2` checkpoint is trained on romanised
phonemes, not Persian script. Passing it Persian text directly produces
near-silence with no error. This module normalises Persian text and
converts it with `mehdi-hf/Homo-GE2PE-Persian-HF` (a `transformers` T5
model) before it reaches PocketTTS, following the usage documented on
https://huggingface.co/mehdi-hf/pocket-tts-farsi-v2.
"""

from __future__ import annotations

import re

from wikiwaves.tts.farsi_normalize import normalize_for_model

G2P_MODEL_ID = "mehdi-hf/Homo-GE2PE-Persian-HF"

# The G2P model emits phonemes using a few overloaded ASCII letters; this
# maps them onto the notation PocketTTS v2 expects.
_TO_PHONEMES = str.maketrans({"/": "a", "a": "A", "@": "?", "$": "S", "c": "C"})

# G2P discards punctuation, so sentence boundaries must be found before
# phonemization, not after.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!؟])\s+")


def split_persian_sentences(text: str) -> list[str]:
    """Split Persian text into sentences on `.`, `!`, and `؟`."""
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    return sentences or [text.strip()]


class FarsiG2P:
    """Lazily-loaded Persian grapheme-to-phoneme converter."""

    def __init__(self, model_id: str = G2P_MODEL_ID):
        try:
            import torch
            from transformers import AutoTokenizer, T5ForConditionalGeneration
        except ImportError as exc:
            raise ImportError(
                "The 'transformers' package is required for PocketTTS Farsi v2."
            ) from exc

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._model = T5ForConditionalGeneration.from_pretrained(model_id).eval()

    def phonemize(self, text: str) -> str:
        """Convert Persian text to the phoneme string PocketTTS v2 expects."""
        text = normalize_for_model(text)
        text = text.replace("؟", "").replace("?", "")
        if not text.strip():
            return ""
        enc = self._tokenizer([text], add_special_tokens=False, return_tensors="pt")
        with self._torch.no_grad():
            out = self._model.generate(**enc, num_beams=5, max_length=512, early_stopping=True)
        raw = self._tokenizer.batch_decode(out, skip_special_tokens=True)[0].strip()
        # G2P marks ezafe with "1" so a chunker can keep noun phrases
        # together. PocketTTS itself has no entry for that digit — strip it
        # before the model sees the text (model-card usage).
        return raw.translate(_TO_PHONEMES).replace("1", "")
