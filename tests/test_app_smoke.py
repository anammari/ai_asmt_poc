import tempfile
import unittest
from pathlib import Path

import app


class TestAppSmoke(unittest.TestCase):
    def test_sanitize_text_for_tts(self):
        raw = "*soft* [pause] # Heading [whisper] hello   world"
        cleaned = app._sanitize_text_for_tts(raw)
        self.assertNotIn("*", cleaned)
        self.assertNotIn("#", cleaned)
        self.assertNotIn("[", cleaned)
        self.assertIn("<<SILENCE700MS>>", cleaned)
        self.assertIn("hello world", cleaned)

    def test_sanitize_text_for_tts_duration_tag(self):
        raw = "Intro [pause:2s] outro"
        cleaned = app._sanitize_text_for_tts(raw)
        self.assertIn("<<SILENCE2000MS>>", cleaned)

    def test_truncate_for_duration(self):
        long_text = " ".join(["word"] * 500)
        trimmed = app._truncate_for_duration(long_text, max_minutes=1, words_per_minute=90)
        self.assertLessEqual(len(trimmed.split()), 90)

    def test_resolve_output_path_creates_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = app._resolve_output_path(str(Path(tmp) / "nested"), "demo")
            self.assertTrue(out.parent.exists())
            self.assertEqual(out.suffix, ".wav")


if __name__ == "__main__":
    unittest.main()
