import unittest
from unittest.mock import patch

import app


class TestAppSmoke(unittest.TestCase):
    def test_sanitize_text_converts_default_pause(self):
        raw = "*soft* [pause] # Heading [whisper] hello   world"
        cleaned = app.sanitize_text(raw)
        self.assertNotIn("*", cleaned)
        self.assertNotIn("[", cleaned)
        self.assertIn("<<SILENCE1500MS>>", cleaned)
        self.assertIn("<<SILENCE1000MS>>", cleaned)
        self.assertIn("Heading", cleaned)

    def test_sanitize_text_converts_duration_pause(self):
        raw = "Intro [pause:2s] outro"
        cleaned = app.sanitize_text(raw)
        self.assertIn("<<SILENCE2000MS>>", cleaned)

    def test_sanitize_text_clamps_pause_bounds(self):
        cleaned = app.sanitize_text("a [pause:0s] b [pause:99s] c")
        self.assertIn("<<SILENCE200MS>>", cleaned)
        self.assertIn("<<SILENCE5000MS>>", cleaned)

    def test_resolve_user_prompt_uses_arabic_typed_input(self):
        text, mode = app._resolve_user_prompt(
            selected_language="Arabic (العربية)",
            arabic_prompt="  صِف لي صوتاً هادئاً للمطر  ",
            audio_file=None,
        )
        self.assertEqual(mode, "typed")
        self.assertEqual(text, "صِف لي صوتاً هادئاً للمطر")

    def test_resolve_user_prompt_uses_stt_for_english(self):
        class _Audio:
            @staticmethod
            def getvalue():
                return b"audio-bytes"

        with patch("app.create_stt_pipeline", return_value="pipe"), patch("app.transcribe", return_value="spoken text") as mock_transcribe:
            text, mode = app._resolve_user_prompt(
                selected_language="English",
                arabic_prompt="",
                audio_file=_Audio(),
            )
        self.assertEqual(mode, "transcribed")
        self.assertEqual(text, "spoken text")
        mock_transcribe.assert_called_once_with("pipe", b"audio-bytes", language="English")

    def test_get_available_voices_for_arabic_whispering(self):
        voices = app.get_available_voices("Arabic (العربية)", "Whispering")
        self.assertIn("ASMR 1 (Arabic Female)", voices)
        self.assertIn("ASMR 2 (Arabic Female)", voices)
        self.assertIn("ASMR 3 (Arabic Female)", voices)
        self.assertNotIn("ASMR 4 (Arabic Female)", voices)
        self.assertNotIn("ASMR 5 (Arabic Female)", voices)
        self.assertTrue(all(v.startswith("fish_") for v in voices.values()))

    def test_get_available_voices_for_arabic_soft_spoken(self):
        voices = app.get_available_voices("Arabic (العربية)", "Soft Spoken")
        self.assertIn("ASMR 4 (Arabic Female)", voices)
        self.assertIn("ASMR 5 (Arabic Female)", voices)
        self.assertNotIn("ASMR 1 (Arabic Female)", voices)
        self.assertNotIn("ASMR 2 (Arabic Female)", voices)
        self.assertNotIn("ASMR 3 (Arabic Female)", voices)

    def test_get_available_voices_for_english_keeps_tone_filter(self):
        voices = app.get_available_voices("English", "Whispering")
        self.assertIn("Nicole (American Female)", voices)
        self.assertNotIn("Bella (American Female)", voices)


if __name__ == "__main__":
    unittest.main()
