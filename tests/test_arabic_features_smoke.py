import unittest
from unittest.mock import patch, MagicMock
import numpy as np

import app
import llm
import tts


class TestArabicFeaturesSmoke(unittest.TestCase):

    @patch('llm._call_gemini')
    def test_arabic_llm_prompt_uses_selected_language(self, mock_gemini):
        mock_gemini.return_value = "Mocked Arabic Script"

        llm.rewrite_script(
            context_text="",
            user_prompt="test",
            provider="gemini",
            language="Arabic (العربية)"
        )

        self.assertIsNotNone(mock_gemini.call_args)
        args, _ = mock_gemini.call_args
        system_msg = args[0]
        self.assertIn("Language rule: Write the spoken script in Arabic (العربية) only.", system_msg)

    def test_arabic_sanitize_markers(self):
        raw_text = "مرحباً [همس] كيف حالك [تنفس] ؟"
        cleaned = app.sanitize_text(raw_text)

        self.assertEqual(cleaned.count("<<SILENCE1000MS>>"), 2)
        self.assertNotIn("[همس]", cleaned)
        self.assertNotIn("[تنفس]", cleaned)
        self.assertIn("مرحباً", cleaned)

    @patch("tts.generate_fish_audio")
    def test_tts_routes_arabic_voice_to_fish(self, mock_generate_fish_audio):
        mock_generate_fish_audio.return_value = np.zeros(24000, dtype=np.float32)
        dummy_pipeline = MagicMock()
        dummy_pipeline.return_value = iter([(None, None, np.array([0.1, 0.2, 0.3], dtype=np.float32))])

        out_bytes = tts.synthesize(
            pipeline=dummy_pipeline,
            text="مرحباً بكم",
            voices=["fish_0de68eaa0cc5438389b82bba728c8e39"]
        )

        mock_generate_fish_audio.assert_called_once()
        dummy_pipeline.assert_not_called()
        self.assertIsInstance(out_bytes, bytes)
        self.assertTrue(out_bytes.startswith(b"RIFF"))


if __name__ == "__main__":
    unittest.main()
