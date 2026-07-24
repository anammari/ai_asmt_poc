import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import tts


class TestTtsSmoke(unittest.TestCase):
    def test_expected_exports_exist(self):
        self.assertTrue(callable(getattr(tts, "create_tts_pipeline", None)))
        self.assertTrue(callable(getattr(tts, "synthesize", None)))

    def test_create_tts_pipeline_returns_config(self):
        with patch.dict(os.environ, {"TTS_ENGINE": "edge-tts"}, clear=False), patch.object(
            tts, "EDGE_TTS_AVAILABLE", True
        ):
            pipeline = tts.create_tts_pipeline(voice="en-US-AnaNeural")

        self.assertEqual(pipeline["engine"], "edge-tts")
        self.assertEqual(pipeline["voice"], "en-US-AnaNeural")
        self.assertIn("speed", pipeline)
        self.assertIn("lang_code", pipeline)

    def test_synthesize_returns_bytes_with_mocked_kokoro(self):
        def fake_generate_speech_kokoro(
            text: str,
            output_path: str,
            voice: str = "af_nicole",
            lang_code: str = "a",
            speed: float = 0.75,
        ) -> str:
            with open(output_path, "wb") as f:
                f.write(b"RIFFFAKEWAV")
            return output_path

        pipeline = {
            "engine": "kokoro",
            "voice": "af_nicole",
            "speed": 0.75,
            "lang_code": "a",
        }

        with patch.object(tts, "generate_speech_kokoro", side_effect=fake_generate_speech_kokoro):
            wav_bytes = tts.synthesize(pipeline, "soft spoken sample")

        self.assertIsInstance(wav_bytes, bytes)
        self.assertTrue(wav_bytes.startswith(b"RIFF"))

    def test_synthesize_uses_provided_output_path(self):
        def fake_generate_speech_kokoro(
            text: str,
            output_path: str,
            voice: str = "af_nicole",
            lang_code: str = "a",
            speed: float = 0.75,
        ) -> str:
            with open(output_path, "wb") as f:
                f.write(b"RIFFOUT")
            return output_path

        pipeline = {
            "engine": "kokoro",
            "voice": "af_nicole",
            "speed": 0.75,
            "lang_code": "a",
        }

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            output_path = tmp.name

        try:
            with patch.object(
                tts, "generate_speech_kokoro", side_effect=fake_generate_speech_kokoro
            ):
                wav_bytes = tts.synthesize(
                    pipeline,
                    "soft spoken sample",
                    output_path=output_path,
                )

            self.assertTrue(os.path.exists(output_path))
            self.assertEqual(wav_bytes, b"RIFFOUT")
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    def test_kokoro_generation_handles_silence_markers(self):
        class FakePipeline:
            def __init__(self, lang_code="a"):
                self.lang_code = lang_code

            def __call__(self, text, voice="af_nicole", speed=0.75, split_pattern=r"\n+"):
                # Return one short chunk for each spoken segment.
                yield (None, None, np.array([0.1, 0.2, 0.3], dtype=np.float32))

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            output_path = tmp.name

        try:
            with patch.object(tts, "KOKORO_AVAILABLE", True), patch.object(
                tts, "KPipeline", FakePipeline
            ):
                tts.generate_speech_kokoro(
                    text="hello <<SILENCE500MS>> world",
                    output_path=output_path,
                    voice="af_nicole",
                    speed=0.75,
                )

            self.assertTrue(os.path.exists(output_path))
            self.assertGreater(os.path.getsize(output_path), 0)
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)


if __name__ == "__main__":
    unittest.main()
