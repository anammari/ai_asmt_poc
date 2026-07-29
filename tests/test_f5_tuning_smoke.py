# tests/test_f5_tuning_smoke.py
import importlib
import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import MagicMock, patch

import numpy as np
import soundfile as sf

import tts


def _write_wav(path: str, seconds: float, sr: int = 24000, channels: int = 1) -> None:
    frames = int(seconds * sr)
    data = np.zeros((frames, channels), dtype=np.float32).squeeze()
    sf.write(path, data, sr, subtype="PCM_16")


class TestF5EnvTuning(unittest.TestCase):
    def test_defaults_match_stock_profile(self):
        # Stock F5 profile proven stable with the Arabic voice; tuned values
        # are applied via env vars only after f5_asmr_inference.py --sweep.
        self.assertEqual(tts._F5_NFE_STEP, 32)
        self.assertEqual(tts._F5_CFG_STRENGTH, 2.0)
        self.assertEqual(tts._F5_SWAY_SAMPLING_COEF, -1.0)
        self.assertEqual(tts._F5_TARGET_RMS, 0.1)

    def test_env_overrides_applied_on_reload(self):
        env = {
            "F5_NFE_STEP": "50",
            "F5_CFG_STRENGTH": "1.8",
            "F5_SWAY_SAMPLING_COEF": "-1.0",
            "ARABIC_REF_AUDIO": "input/custom_ref.wav",
        }
        with patch.dict(os.environ, env, clear=False):
            reloaded = importlib.reload(tts)
        try:
            self.assertEqual(reloaded._F5_NFE_STEP, 50)
            self.assertEqual(reloaded._F5_CFG_STRENGTH, 1.8)
            self.assertEqual(reloaded._F5_SWAY_SAMPLING_COEF, -1.0)
            self.assertEqual(reloaded._ARABIC_REF_AUDIO, "input/custom_ref.wav")
        finally:
            importlib.reload(tts)

    def test_invalid_env_values_fall_back_to_defaults(self):
        with patch.dict(os.environ, {"F5_NFE_STEP": "not-a-number"}, clear=False):
            reloaded = importlib.reload(tts)
        try:
            self.assertEqual(reloaded._F5_NFE_STEP, 32)
        finally:
            importlib.reload(tts)


class TestRefAudioValidation(unittest.TestCase):
    def setUp(self):
        tts._REF_AUDIO_WARNED = False

    def test_spec_compliant_clip_has_no_warnings(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ok.wav")
            _write_wav(path, seconds=8.0, sr=24000, channels=1)
            info = tts._validate_ref_audio(path)
        self.assertEqual(info["warnings"], [])
        self.assertEqual(info["duration_s"], 8.0)
        self.assertEqual(info["channels"], 1)

    def test_off_spec_clip_warns_duration_stereo_rate(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad.wav")
            _write_wav(path, seconds=13.0, sr=16000, channels=2)
            with redirect_stdout(io.StringIO()) as buf:
                info = tts._validate_ref_audio(path)
        self.assertEqual(len(info["warnings"]), 3)
        self.assertIn("prepare_ref_audio.py", buf.getvalue())

    def test_warnings_print_only_once_per_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad.wav")
            _write_wav(path, seconds=2.0, sr=16000, channels=2)
            with redirect_stdout(io.StringIO()) as first:
                tts._validate_ref_audio(path)
            with redirect_stdout(io.StringIO()) as second:
                tts._validate_ref_audio(path)
        self.assertIn("[Warning]", first.getvalue())
        self.assertEqual(second.getvalue(), "")


class TestRefTextLoading(unittest.TestCase):
    def tearDown(self):
        tts._ARABIC_REF_TEXT_CACHE = None
        tts._ARABIC_REF_TEXT_PATH = os.getenv("ARABIC_REF_TEXT_PATH", "input/test_ahmad_2.txt")

    def test_loads_text_from_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ref.txt")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("نص مرجعي للاختبار\n")
            tts._ARABIC_REF_TEXT_CACHE = None
            tts._ARABIC_REF_TEXT_PATH = path
            self.assertEqual(tts._load_arabic_ref_text(), "نص مرجعي للاختبار")

    def test_falls_back_to_inline_when_file_missing(self):
        tts._ARABIC_REF_TEXT_CACHE = None
        tts._ARABIC_REF_TEXT_PATH = "input/does_not_exist.txt"
        self.assertEqual(tts._load_arabic_ref_text(), tts._ARABIC_REF_TEXT)


class TestArabicPreprocessingHook(unittest.TestCase):
    def setUp(self):
        tts._PREPROCESS_WARNED = False

    def test_applies_normalize_then_diacritize(self):
        fake = MagicMock()
        fake.normalize_arabic.return_value = "normalized"
        fake.diacritize.return_value = "diacritized"
        with patch.dict("sys.modules", {"preprocess_arabic": fake}):
            out = tts._apply_arabic_preprocessing("raw text")
        self.assertEqual(out, "diacritized")
        fake.normalize_arabic.assert_called_once_with("raw text")
        fake.diacritize.assert_called_once_with("normalized")

    def test_missing_diacritizer_returns_normalized_text(self):
        fake = MagicMock()
        fake.normalize_arabic.return_value = "normalized"
        fake.diacritize.side_effect = ImportError("mishkal missing")
        with patch.dict("sys.modules", {"preprocess_arabic": fake}):
            out = tts._apply_arabic_preprocessing("raw text")
        self.assertEqual(out, "normalized")

    def test_missing_module_returns_raw_text(self):
        with patch.dict("sys.modules", {"preprocess_arabic": None}):
            out = tts._apply_arabic_preprocessing("raw text")
        self.assertEqual(out, "raw text")

    def test_empty_text_passthrough(self):
        self.assertEqual(tts._apply_arabic_preprocessing("   "), "   ")


class TestF5OutputValidation(unittest.TestCase):
    def test_rejects_empty(self):
        with self.assertRaises(RuntimeError):
            tts._validate_f5_output(np.array([], dtype=np.float32))

    def test_rejects_non_finite(self):
        with self.assertRaises(RuntimeError):
            tts._validate_f5_output(np.array([0.0, np.inf], dtype=np.float32))

    def test_clamps_clipping(self):
        loud = np.full(5000, 1.5, dtype=np.float32)
        with redirect_stdout(io.StringIO()):
            out = tts._validate_f5_output(loud)
        self.assertLessEqual(float(np.max(np.abs(out))), 1.0)


class TestGenerateF5AudioParams(unittest.TestCase):
    """Verifies tuned params reach infer_process (model/vocoder mocked out)."""

    def setUp(self):
        tts._REF_AUDIO_WARNED = True  # keep test output quiet
        self.tmp = tempfile.TemporaryDirectory()
        self.ref_path = os.path.join(self.tmp.name, "ref.wav")
        _write_wav(self.ref_path, seconds=8.0, sr=24000, channels=1)

    def tearDown(self):
        self.tmp.cleanup()

    def _run_with_mocks(self, **kwargs):
        fake_wav = np.zeros(2400, dtype=np.float32)
        kwargs.setdefault("use_preprocessing", False)
        # autospec keeps the real infer_process signature so tts.py's
        # signature-based param filtering works against the mock.
        with patch("f5_tts.infer.utils_infer.infer_process", autospec=True,
                   return_value=(fake_wav, 24000, None)) as mock_infer, \
             patch.object(tts, "_load_silma_model", return_value=MagicMock()), \
             patch.object(tts, "_load_silma_vocoder", return_value=MagicMock()):
            out = tts.generate_f5_audio(
                "نص للاختبار",
                ref_audio=self.ref_path,
                ref_text="نص مرجعي",
                **kwargs,
            )
        self.assertIsInstance(out, np.ndarray)
        return mock_infer

    def test_env_defaults_forwarded(self):
        mock_infer = self._run_with_mocks()
        _, call_kwargs = mock_infer.call_args
        self.assertEqual(call_kwargs["nfe_step"], tts._F5_NFE_STEP)
        self.assertEqual(call_kwargs["cfg_strength"], tts._F5_CFG_STRENGTH)
        self.assertEqual(call_kwargs["sway_sampling_coef"], tts._F5_SWAY_SAMPLING_COEF)
        self.assertEqual(call_kwargs["target_rms"], tts._F5_TARGET_RMS)

    def test_explicit_legacy_overrides_forwarded(self):
        mock_infer = self._run_with_mocks(
            nfe_step=32, cfg_strength=2.0, sway_sampling_coef=-1.0
        )
        _, call_kwargs = mock_infer.call_args
        self.assertEqual(call_kwargs["nfe_step"], 32)
        self.assertEqual(call_kwargs["sway_sampling_coef"], -1.0)

    def test_preprocessing_applies_to_gen_text_only_not_ref_text(self):
        # The reference transcript must reach F5 EXACTLY as provided:
        # algorithmically altering it breaks F5's character-to-phoneme
        # alignment with the ref audio (root cause of the silent-output bug).
        fake = MagicMock()
        fake.normalize_arabic.side_effect = lambda t: f"N({t})"
        fake.diacritize.side_effect = lambda t: f"D({t})"
        with patch.dict("sys.modules", {"preprocess_arabic": fake}):
            mock_infer = self._run_with_mocks(use_preprocessing=True)
        _, call_kwargs = mock_infer.call_args
        self.assertEqual(call_kwargs["gen_text"], "D(N(نص للاختبار))")
        self.assertEqual(call_kwargs["ref_text"], "نص مرجعي")  # verbatim

    def test_missing_ref_audio_raises(self):
        with self.assertRaises(FileNotFoundError):
            tts.generate_f5_audio("نص", ref_audio="no/such/file.wav")


if __name__ == "__main__":
    unittest.main()
