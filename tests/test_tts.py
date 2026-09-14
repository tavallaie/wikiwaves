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
        self.assertIsNone(engine._text_prep)

    def test_synthesize_passes_frames_after_eos(self):
        from wikiwaves.tts.pocket import PocketTTSEngine

        engine = PocketTTSEngine(frames_after_eos=0)
        engine.synthesize("Hello", voice="alba")
        self.assertEqual(engine._tts.generate_calls[0][2], 0)


class FakeG2P:
    """Stand-in for FarsiG2P that never downloads weights."""

    def __init__(self):
        self.calls: list[str] = []

    def phonemize(self, text: str) -> str:
        self.calls.append(text)
        # Keep the ezafe marker "1" out; strip to a simple phoneme stub.
        return f"ph:{text[:20]}"


class TestFarsiV2G2P(unittest.TestCase):
    def setUp(self):
        self._patcher = _install_fake_pocket_tts()
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()

    def test_v2_config_enables_g2p(self):
        from wikiwaves.tts.pocket import PocketTTSEngine

        fake = FakeG2P()
        with mock.patch("wikiwaves.tts.farsi_g2p.FarsiG2P", return_value=fake):
            engine = PocketTTSEngine(
                config="hf://mehdi-hf/pocket-tts-farsi-v2/model.yaml",
            )
        self.assertIs(engine._text_prep, fake)

    def test_non_v2_config_uses_community_passthrough(self):
        from wikiwaves.tts.pocket import PocketTTSEngine

        engine = PocketTTSEngine(config="hf://mehdi-hf/pocket-tts-farsi/farsi.yaml")
        self.assertIsNone(engine._text_prep)
        self.assertEqual(engine.profile.id, "community")

    def test_no_config_uses_official(self):
        from wikiwaves.tts.pocket import PocketTTSEngine

        engine = PocketTTSEngine()
        self.assertIsNone(engine._text_prep)
        self.assertEqual(engine.profile.id, "official")

    def test_explicit_profile_overrides_config_match(self):
        from wikiwaves.tts.pocket import PocketTTSEngine

        # farsi-v2 config URL, but force official text path (no G2P)
        engine = PocketTTSEngine(
            config="hf://mehdi-hf/pocket-tts-farsi-v2/model.yaml",
            profile="community",
        )
        self.assertEqual(engine.profile.id, "community")
        self.assertIsNone(engine._text_prep)

    def test_synthesize_phonemizes_each_sentence(self):
        from wikiwaves.tts.pocket import PocketTTSEngine

        fake = FakeG2P()
        with mock.patch("wikiwaves.tts.farsi_g2p.FarsiG2P", return_value=fake):
            engine = PocketTTSEngine(
                config="hf://mehdi-hf/pocket-tts-farsi-v2/model.yaml",
                frames_after_eos=0,
            )
        text = "سلام، حال شما چطور است؟ امیدوارم روز خوبی داشته باشید."
        wav = engine.synthesize(text, voice="prompt.wav", silence_duration=0.0)
        self.assertEqual(wav.shape[0], 1)
        self.assertEqual(wav.dtype, np.float32)
        self.assertEqual(len(fake.calls), 2)
        self.assertEqual(len(engine._tts.generate_calls), 2)
        # generate_audio receives phonemes, not Persian script
        self.assertTrue(engine._tts.generate_calls[0][1].startswith("ph:"))
        self.assertTrue(engine._tts.generate_calls[1][1].startswith("ph:"))

    def test_community_config_rejects_catalog_voice(self):
        from wikiwaves.tts.pocket import PocketTTSEngine

        fake = FakeG2P()
        with mock.patch("wikiwaves.tts.farsi_g2p.FarsiG2P", return_value=fake):
            engine = PocketTTSEngine(
                config="hf://mehdi-hf/pocket-tts-farsi-v2/model.yaml",
            )
        with self.assertRaises(ValueError) as ctx:
            engine.get_voice_style("alba")
        self.assertIn("voice wav", str(ctx.exception))

    def test_farsi_v2_applies_model_card_defaults(self):
        from wikiwaves.tts.pocket import PocketTTSEngine

        fake = FakeG2P()
        with mock.patch("wikiwaves.tts.farsi_g2p.FarsiG2P", return_value=fake):
            engine = PocketTTSEngine(
                config="hf://mehdi-hf/pocket-tts-farsi-v2/model.yaml",
                sampler_decode_steps=5,  # runner default
                # eos_threshold / frames / temp left at PocketTTSEngine defaults
            )
        self.assertEqual(engine._tts.sampler_decode_steps, 1)
        self.assertEqual(engine._tts.eos_threshold, -2.0)
        self.assertEqual(engine.frames_after_eos, 0)
        self.assertEqual(engine._tts.temp, 0.3)

    def test_missing_transformers_raises_runtime_error(self):
        from wikiwaves.tts.pocket import PocketTTSEngine

        with mock.patch(
            "wikiwaves.tts.farsi_g2p.FarsiG2P",
            side_effect=ImportError("No module named 'transformers'"),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                PocketTTSEngine(config="hf://mehdi-hf/pocket-tts-farsi-v2/model.yaml")
        self.assertIn("transformers", str(ctx.exception))

    def test_split_persian_sentences(self):
        from wikiwaves.tts.farsi_g2p import split_persian_sentences

        parts = split_persian_sentences("اول. دوم! سوم؟ چهارم")
        self.assertEqual(parts, ["اول.", "دوم!", "سوم؟", "چهارم"])


class TestPocketProfiles(unittest.TestCase):
    def test_resolve_explicit_profile(self):
        from wikiwaves.tts.pocket_profiles import resolve_profile

        p = resolve_profile(profile="farsi-v2", config=None, language="english")
        self.assertEqual(p.id, "farsi-v2")

    def test_resolve_config_auto_detects_farsi_v2(self):
        from wikiwaves.tts.pocket_profiles import resolve_profile

        p = resolve_profile(config="hf://mehdi-hf/pocket-tts-farsi-v2/model.yaml")
        self.assertEqual(p.id, "farsi-v2")

    def test_resolve_unmatched_config_is_community(self):
        from wikiwaves.tts.pocket_profiles import resolve_profile

        p = resolve_profile(config="hf://someone/other-model/model.yaml")
        self.assertEqual(p.id, "community")

    def test_resolve_language_is_official(self):
        from wikiwaves.tts.pocket_profiles import resolve_profile

        p = resolve_profile(language="french_24l")
        self.assertEqual(p.id, "official")

    def test_resolve_unknown_profile_raises(self):
        from wikiwaves.tts.pocket_profiles import resolve_profile

        with self.assertRaises(ValueError):
            resolve_profile(profile="nope")


class TestFarsiNormalize(unittest.TestCase):
    def test_normalize_folds_arabic_yeh_and_spells_digits(self):
        from wikiwaves.tts.farsi_normalize import normalize_for_model

        out = normalize_for_model("تا سال ۲۰۳۰")
        self.assertIn("هزار", out)
        self.assertNotIn("ي", out)

    def test_is_phonemic_passes_through(self):
        from wikiwaves.tts.farsi_normalize import normalize_for_model

        phonemes = "salAm hAle SomA Cetor ?ast"
        self.assertEqual(normalize_for_model(phonemes), phonemes)


class TestFarsiG2PPhonemize(unittest.TestCase):
    def test_phonemize_strips_ezafe_marker(self):
        """G2P may emit '1' for ezafe; PocketTTS must not see that digit."""
        from contextlib import nullcontext

        class FakeTok:
            def __call__(self, texts, add_special_tokens=False, return_tensors=None):
                return {"input_ids": [[1, 2]]}

            def batch_decode(self, out, skip_special_tokens=True):
                return ["?eqtesAde1 ?AmrikA"]

        class FakeModel:
            def generate(self, **kwargs):
                return [[0]]

        from wikiwaves.tts.farsi_g2p import FarsiG2P

        g2p = object.__new__(FarsiG2P)
        g2p._torch = types.SimpleNamespace(no_grad=nullcontext)
        g2p._tokenizer = FakeTok()
        g2p._model = FakeModel()
        out = g2p.phonemize("اقتصاد آمریکا")
        self.assertNotIn("1", out)
        self.assertIn("?eqtesAde", out)


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

    def test_run_accepts_txt_file_path(self):
        class FakeEngine:
            sample_rate = 24000

            def get_voice_style(self, voice_name):
                return {"voice": voice_name}

            def synthesize(self, text, voice=None, **kwargs):
                self.last_text = text
                return np.ones((1, 240), dtype=np.float32)

            def save_audio(self, wav, path):
                import soundfile as sf

                path = Path(path)
                path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(str(path), wav.T, self.sample_rate)

        engine = FakeEngine()
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "pockettts-farsi-script.txt"
            script.write_text(
                "به ویکی‌ویوز خوش آمدید.\nامروز پنج رویداد را مرور می‌کنیم.\n",
                encoding="utf-8",
            )
            with mock.patch(
                "wikiwaves.tts.runner.create_engine",
                return_value=engine,
            ):
                from wikiwaves.tts.runner import run

                paths = run(date_str=str(script), backend="pockettts")

            self.assertTrue(paths)
            self.assertTrue(paths[0].exists())
            self.assertIn("ویکی‌ویوز", engine.last_text)

    def test_read_plain_txt_keeps_first_line(self):
        from wikiwaves.tts.runner import read_script_txt

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "notes.txt"
            path.write_text("خط اول\nخط دوم\n", encoding="utf-8")
            seg = read_script_txt(path)
        self.assertIn("خط اول", seg.text)
        self.assertIn("خط دوم", seg.text)


if __name__ == "__main__":
    unittest.main()
