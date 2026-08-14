import unittest
from unittest.mock import patch

import numpy as np

import tts


class TestTtsRegressions(unittest.TestCase):
    def test_synthesize_rejects_non_string_text(self):
        with self.assertRaises(TypeError):
            tts.synthesize(None)

    def test_synthesize_rejects_empty_text(self):
        with self.assertRaises(ValueError):
            tts.synthesize("   ")

    def test_synthesize_concatenates_paragraph_audio(self):
        fake_audio = np.array([0.1, -0.1, 0.2], dtype=np.float32)
        with patch.object(tts, "generate_fish_audio", return_value=fake_audio):
            wav_bytes = tts.synthesize("first\n\nsecond", voices=["fish_0de68eaa0cc5438389b82bba728c8e39"])

        self.assertTrue(wav_bytes.startswith(b"RIFF"))


if __name__ == "__main__":
    unittest.main()
