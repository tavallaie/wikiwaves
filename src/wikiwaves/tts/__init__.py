"""TTS module — speech synthesis (Supertonic and PocketTTS)."""

from wikiwaves.tts.engine import TTSEngine, create_engine
from wikiwaves.tts.pocket import PocketTTSEngine

__all__ = ["TTSEngine", "PocketTTSEngine", "create_engine"]
