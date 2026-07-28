import unittest

import numpy as np

import tts


class _FakePipeline:
    def __call__(self, text, voice="af_bella", speed=0.85, split_pattern=r"\n+"):
        yield None, None, np.array([0.1, -0.1, 0.2], dtype=np.float32)


class _FakePipelineWithNoneChunk:
    def __call__(self, text, voice="af_bella", speed=0.85, split_pattern=r"\n+"):
        yield None, None, None


class TestTtsRegressions(unittest.TestCase):
    def test_synthesize_rejects_non_string_text(self):
        with self.assertRaises(TypeError):
            tts.synthesize(_FakePipeline(), None)

    def test_synthesize_rejects_empty_text(self):
        with self.assertRaises(ValueError):
            tts.synthesize(_FakePipeline(), "   ")

    def test_synthesize_rejects_none_audio_chunks(self):
        with self.assertRaises(RuntimeError):
            tts.synthesize(_FakePipelineWithNoneChunk(), "hello")


if __name__ == "__main__":
    unittest.main()
