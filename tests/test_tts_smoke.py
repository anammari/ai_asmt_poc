import unittest
from unittest.mock import patch

import numpy as np

import tts


class TestTtsSmoke(unittest.TestCase):
    def test_expected_exports_exist(self):
        self.assertTrue(callable(getattr(tts, "create_tts_pipeline", None)))
        self.assertTrue(callable(getattr(tts, "synthesize", None)))

    def test_create_tts_pipeline_returns_none_when_kokoro_missing(self):
        with patch.object(tts, "KOKORO_AVAILABLE", False):
            pipeline = tts.create_tts_pipeline()
        self.assertIsNone(pipeline)

    def test_create_tts_pipeline_uses_kpipeline_when_available(self):
        fake_instance = object()
        with patch.object(tts, "KOKORO_AVAILABLE", True), patch.object(tts, "KPipeline", return_value=fake_instance) as mock_ctor:
            pipeline = tts.create_tts_pipeline(lang_code="a")
        self.assertIs(pipeline, fake_instance)
        mock_ctor.assert_called_once_with(lang_code="a")

    def test_synthesize_returns_wav_bytes_with_mocked_pipeline(self):
        class FakePipeline:
            def __call__(self, text, voice="af_bella", speed=0.85, split_pattern=r"\n+"):
                yield (None, None, np.array([0.1, 0.2, 0.3], dtype=np.float32))

        wav_bytes = tts.synthesize(
            FakePipeline(),
            "hello <<SILENCE500MS>> world",
            voices=["af_bella"],
        )

        self.assertIsInstance(wav_bytes, bytes)
        self.assertTrue(wav_bytes.startswith(b"RIFF"))

    def test_synthesize_raises_when_pipeline_missing(self):
        with patch.object(tts, "KOKORO_AVAILABLE", False):
            with self.assertRaises(ImportError):
                tts.synthesize(None, "hello world")


if __name__ == "__main__":
    unittest.main()
