import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from finetuning import build_dataset as builder


class TestBuildDataset(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.ingested_dir = self.tmpdir.name
        self._write_json(
            "syria_sample.json",
            {
                "dialect": "syria",
                "video_id": "sample1",
                "title": "همسات المساء",
                "transcript": "[pause:2s] هلا فيك.\nنحنا هلق بالليل.\n",
            },
        )
        self._write_json(
            "egypt_sample.json",
            {
                "dialect": "egypt",
                "video_id": "sample2",
                "title": "تهدئة قبل النوم",
                "transcript": "[pause:2s] أهلاً بيك.\nخد نفسك.\n",
            },
        )
        self._write_json(
            "empty_title.json",
            {
                "dialect": "syria",
                "video_id": "sample3",
                "title": "   ",
                "transcript": "transcript without title",
            },
        )
        self._write_json(
            "empty_transcript.json",
            {
                "dialect": "syria",
                "video_id": "sample4",
                "title": "title without transcript",
                "transcript": "",
            },
        )

    def tearDown(self):
        self.tmpdir.cleanup()

    def _write_json(self, filename: str, data: dict) -> None:
        path = os.path.join(self.ingested_dir, filename)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False)

    def _load_jsonl(self, path: str) -> list[dict]:
        records = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def test_syria_chat_structure(self):
        out_path = os.path.join(self.tmpdir.name, "syria_out.jsonl")
        code = builder.main([
            "--dialect", "syria",
            "--ingestion-dir", self.ingested_dir,
            "--out", out_path,
        ])
        self.assertEqual(code, 0)

        records = self._load_jsonl(out_path)
        # Only the valid syria sample should survive the missing-title/transcript filters.
        self.assertEqual(len(records), 1)

        record = records[0]
        self.assertIn("messages", record)
        self.assertEqual(len(record["messages"]), 3)

        for msg in record["messages"]:
            self.assertIn("role", msg)
            self.assertIsInstance(msg["content"], str)

        self.assertEqual(record["messages"][0]["role"], "system")
        self.assertEqual(record["messages"][0]["content"], builder.SYSTEM_PROMPT)
        self.assertEqual(record["messages"][1]["role"], "user")
        self.assertIn("همسات المساء", record["messages"][1]["content"])
        self.assertTrue(
            record["messages"][1]["content"].startswith("User Instructions / Topic:")
        )
        self.assertEqual(record["messages"][2]["role"], "assistant")
        self.assertIn("نحنا هلق بالليل", record["messages"][2]["content"])

    def test_egypt_chat_structure(self):
        out_path = os.path.join(self.tmpdir.name, "egypt_out.jsonl")
        code = builder.main([
            "--dialect", "egypt",
            "--ingestion-dir", self.ingested_dir,
            "--out", out_path,
        ])
        self.assertEqual(code, 0)

        records = self._load_jsonl(out_path)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["messages"][1]["role"], "user")
        self.assertIn("تهدئة قبل النوم", records[0]["messages"][1]["content"])
        self.assertIn("أهلاً بيك", records[0]["messages"][2]["content"])

    def test_invalid_dialect(self):
        with self.assertRaises(SystemExit):
            builder.main(["--dialect", "lebanon"])


if __name__ == "__main__":
    unittest.main()
