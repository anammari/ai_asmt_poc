import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from finetuning import ingest_YT_transcript as ingest


class TestIngestYT(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.output_root = self.tmpdir.name
        self.csv_path = os.path.join(self.output_root, "metadata.csv")
        with open(self.csv_path, "w", encoding="utf-8") as fh:
            fh.write("Dialect,URL\n")
            fh.write(
                "Syria,https://www.youtube.com/watch?v=dQw4w9WgXcQ\n"
            )
            fh.write(
                "Egypt,https://www.youtube.com/watch?v=9bZkp7q19f0\n"
            )
            fh.write("Unknown,https://www.youtube.com/watch?v=abcd1234efg\n")
            fh.write("Syria,\n")

    def tearDown(self):
        self.tmpdir.cleanup()

    @patch.object(ingest, "_fetch_title_ytdlp", side_effect=["Syria Video", "Egypt Video"])
    @patch.object(
        ingest,
        "_fetch_transcript_yta",
        side_effect=["نص سوري هادئ", "نص مصري هادئ"],
    )
    def test_csv_parsing_and_routing(self, mock_transcript, mock_title):
        code = ingest.main([
            "--metadata-csv",
            self.csv_path,
            "--output-root",
            self.output_root,
        ])
        self.assertEqual(code, 0)

        syria_dir = os.path.join(self.output_root, "syria")
        egypt_dir = os.path.join(self.output_root, "egypt")
        self.assertTrue(os.path.isdir(syria_dir))
        self.assertTrue(os.path.isdir(egypt_dir))

        syria_files = [f for f in os.listdir(syria_dir) if f.endswith(".json")]
        egypt_files = [f for f in os.listdir(egypt_dir) if f.endswith(".json")]
        self.assertEqual(len(syria_files), 1)
        self.assertEqual(len(egypt_files), 1)

        with open(os.path.join(syria_dir, syria_files[0]), encoding="utf-8") as fh:
            data = json.load(fh)
        self.assertEqual(data["dialect"], "syria")
        self.assertEqual(data["title"], "Syria Video")
        self.assertEqual(data["transcript"], "نص سوري هادئ")
        self.assertEqual(data["video_id"], "dQw4w9WgXcQ")

        with open(os.path.join(egypt_dir, egypt_files[0]), encoding="utf-8") as fh:
            data = json.load(fh)
        self.assertEqual(data["dialect"], "egypt")
        self.assertEqual(data["title"], "Egypt Video")
        self.assertEqual(data["transcript"], "نص مصري هادئ")
        self.assertEqual(data["video_id"], "9bZkp7q19f0")

    def test_video_id_parser(self):
        self.assertEqual(
            ingest._video_id("https://www.youtube.com/watch?v=AbC123def45"),
            "AbC123def45",
        )
        self.assertEqual(
            ingest._video_id("https://youtu.be/AbC123def45"),
            "AbC123def45",
        )
        self.assertEqual(
            ingest._video_id("https://www.youtube.com/embed/AbC123def45"),
            "AbC123def45",
        )
        self.assertIsNone(ingest._video_id("not a url"))
        self.assertIsNone(ingest._video_id(""))


if __name__ == "__main__":
    unittest.main()
