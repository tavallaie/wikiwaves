"""Tests for the TTS backends."""

from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


class _FakeTensor:
    """Minimal torch-like tensor with detach/cpu/numpy."""

    def __init__(self, data):
        self._data = np.asarray(data, dtype=np.float32)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._data


class FakeTTSModel:
    """Stand-in for pocket_tts.TTSModel that never loads weights."""

    def __init__(self):
        self.sample_rate = 24000
        self.prompt_calls: list[str] = []
        self.generate_calls: list[tuple] = []

    @classmethod
    def load_model(
        cls,
        language=None,
        config=None,
        sampler_decode_steps=1,
        temp=None,
        quantize=False,
        eos_threshold=-4.0,
        **kwargs,
    ):
        inst = cls()
        inst.language = language
        inst.config = config
        inst.sampler_decode_steps = sampler_decode_steps
        inst.temp = temp
        inst.quantize = quantize
        inst.eos_threshold = eos_threshold
        inst.load_kwargs = kwargs
        return inst

    def get_state_for_audio_prompt(self, name):
        self.prompt_calls.append(name)
        return {"voice": name}

    def generate_audio(self, state, text, frames_after_eos=None, copy_state=True):
        self.generate_calls.append((state, text, frames_after_eos))
        return _FakeTensor(np.ones(240, dtype=np.float32))


def _install_fake_pocket_tts():
    mod = types.ModuleType("pocket_tts")
    mod.TTSModel = FakeTTSModel
    return mock.patch.dict(sys.modules, {"pocket_tts": mod})


class TestPocketTTSMissingPackage(unittest.TestCase):
    def test_missing_pocket_tts_raises_runtime_error(self):
        with mock.patch.dict(sys.modules, {"pocket_tts": None}):
            from wikiwaves.tts.pocket import PocketTTSEngine

            with self.assertRaises(RuntimeError) as ctx:
                PocketTTSEngine()
        self.assertIn("pocket-tts", str(ctx.exception))


class TestPocketTTSEngine(unittest.TestCase):
    def setUp(self):
        self._patcher = _install_fake_pocket_tts()
        self._patcher.start()
        from wikiwaves.tts.pocket import PocketTTSEngine

        self.engine = PocketTTSEngine()

    def tearDown(self):
        self._patcher.stop()

    def test_synthesize_converts_1d_to_mono_float32(self):
        wav = self.engine.synthesize("Hello world", voice="alba")
        self.assertIsInstance(wav, np.ndarray)
        self.assertEqual(wav.shape[0], 1)
        self.assertEqual(wav.ndim, 2)
        self.assertEqual(wav.dtype, np.float32)
        self.assertGreater(wav.shape[1], 0)

    def test_empty_text_returns_empty_and_skips_generate(self):
        wav = self.engine.synthesize("   ", voice="alba")
        self.assertEqual(wav.shape, (1, 0))
        self.assertEqual(wav.dtype, np.float32)
        self.assertEqual(self.engine._tts.generate_calls, [])

    def test_get_voice_style_alba_calls_audio_prompt(self):
        style = self.engine.get_voice_style("alba")
        self.assertEqual(style, {"voice": "alba"})
        self.assertIn("alba", self.engine._tts.prompt_calls)

    def test_get_voice_style_passes_local_path(self):
        path = "/tmp/custom_voice.wav"
        style = self.engine.get_voice_style(path)
        self.assertEqual(style, {"voice": path})
        self.assertEqual(self.engine._tts.prompt_calls[-1], path)

    def test_list_voices_includes_alba(self):
        voices = self.engine.list_voices()
        self.assertIn("alba", voices)

    def test_config_is_passed_instead_of_language(self):
        from wikiwaves.tts.pocket import PocketTTSEngine

        engine = PocketTTSEngine(
            config="hf://mehdi-hf/pocket-tts-farsi/farsi.yaml",
            temp=0.3,
            eos_threshold=-2.0,
            frames_after_eos=0,
        )
        self.assertEqual(engine._tts.config, "hf://mehdi-hf/pocket-tts-farsi/farsi.yaml")
        self.assertIsNone(engine._tts.language)
        self.assertEqual(engine._tts.temp, 0.3)
        self.assertEqual(engine._tts.eos_threshold, -2.0)
        self.assertEqual(engine.frames_after_eos, 0)

    def test_synthesize_passes_frames_after_eos(self):
        from wikiwaves.tts.pocket import PocketTTSEngine

        engine = PocketTTSEngine(frames_after_eos=0)
        engine.synthesize("Hello", voice="alba")
        self.assertEqual(engine._tts.generate_calls[0][2], 0)


class TestCreateEngine(unittest.TestCase):
    def setUp(self):
        self._patcher = _install_fake_pocket_tts()
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()

    def test_create_engine_pockettts_returns_pocket_engine(self):
        from wikiwaves.tts.engine import create_engine
        from wikiwaves.tts.pocket import PocketTTSEngine

        engine = create_engine(backend="pockettts")
        self.assertIsInstance(engine, PocketTTSEngine)

    def test_create_engine_supertonic_returns_tts_engine(self):
        from wikiwaves.tts.engine import TTSEngine, create_engine

        fake_tts = mock.Mock()
        fake_tts.sample_rate = 24000
        fake_tts.voice_style_names = ["M1"]
        with mock.patch("wikiwaves.tts.engine._SupertonicTTS", return_value=fake_tts):
            engine = create_engine(backend="supertonic")
        self.assertIsInstance(engine, TTSEngine)

    def test_create_engine_unknown_backend_raises_value_error(self):
        from wikiwaves.tts.engine import create_engine

        with self.assertRaises(ValueError):
            create_engine(backend="nope")


class TestTTSRunner(unittest.TestCase):
    def test_run_pockettts_writes_wav(self):
        class FakeEngine:
            sample_rate = 24000

            def get_voice_style(self, voice_name):
                return {"voice": voice_name}

            def synthesize(self, text, voice=None, **kwargs):
                return np.ones((1, 240), dtype=np.float32)

            def save_audio(self, wav, path):
                import soundfile as sf

                path = Path(path)
                path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(str(path), wav.T, self.sample_rate)

        with tempfile.TemporaryDirectory() as tmp:
            day = Path(tmp) / "2026-05-11"
            day.mkdir()
            (day / "intro.txt").write_text(
                "Episode Intro\n\nHello from the host.\n",
                encoding="utf-8",
            )

            with mock.patch(
                "wikiwaves.tts.runner.create_engine",
                return_value=FakeEngine(),
            ):
                from wikiwaves.tts.runner import run

                paths = run(
                    date_str="2026-05-11",
                    output_dir=tmp,
                    backend="pockettts",
                )

            self.assertTrue(paths)
            self.assertTrue(paths[0].exists())


if __name__ == "__main__":
    unittest.main()
