import unittest
from unittest.mock import patch

import numpy as np

import tts


class TestTtsSmoke(unittest.TestCase):
    def test_expected_exports_exist(self):
        self.assertTrue(callable(getattr(tts, "synthesize", None)))
        self.assertTrue(callable(getattr(tts, "generate_fish_audio", None)))

    def test_synthesize_returns_wav_bytes_with_mocked_fish(self):
        fake_audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        with patch.object(tts, "generate_fish_audio", return_value=fake_audio) as mock_fish:
            wav_bytes = tts.synthesize(
                "hello <<SILENCE500MS>> world",
                voices=["fish_0de68eaa0cc5438389b82bba728c8e39"],
            )

        self.assertIsInstance(wav_bytes, bytes)
        self.assertTrue(wav_bytes.startswith(b"RIFF"))
        mock_fish.assert_called_once()

    def test_synthesize_alternates_voices_across_paragraphs(self):
        fake_audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        voices = [
            "fish_0de68eaa0cc5438389b82bba728c8e39",
            "fish_2689bc84ab944610af10bf64e586684a",
        ]
        with patch.object(tts, "generate_fish_audio", return_value=fake_audio) as mock_fish:
            tts.synthesize("para one\n\npara two\n\npara three", voices=voices)

        self.assertEqual(mock_fish.call_count, 3)
        first_id = mock_fish.call_args_list[0].kwargs["voice_id"]
        second_id = mock_fish.call_args_list[1].kwargs["voice_id"]
        third_id = mock_fish.call_args_list[2].kwargs["voice_id"]
        self.assertEqual(first_id, "0de68eaa0cc5438389b82bba728c8e39")
        self.assertEqual(second_id, "2689bc84ab944610af10bf64e586684a")
        self.assertEqual(third_id, "0de68eaa0cc5438389b82bba728c8e39")


if __name__ == "__main__":
    unittest.main()
