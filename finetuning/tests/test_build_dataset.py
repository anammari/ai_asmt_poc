import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# Import the module itself (not its main) so we can test internals.
import importlib.util
spec = importlib.util.spec_from_file_location(
    "builder", REPO_ROOT / "finetuning" / "build_dataset.py"
)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class TestBuildDatasetAppendOnly(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.ingested_dir = os.path.join(self.tmpdir.name, "ingested")
        self.jsonl_path = os.path.join(self.tmpdir.name, "training.jsonl")
        os.makedirs(self.ingested_dir)

    def tearDown(self):
        self.tmpdir.cleanup()

    def _write_ingested(self, filename: str, dialect: str, title: str, transcript: str):
        path = os.path.join(self.ingested_dir, filename)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({
                "dialect": dialect,
                "video_id": filename.replace(".json", ""),
                "url": f"https://youtube.com/watch?v={filename.replace('.json', '')}",
                "title": title,
                "transcript": transcript,
                "ingested_at": "2025-01-01T00:00:00",
            }, fh, ensure_ascii=False)

    def _write_jsonl(self, records: list[dict]):
        with open(self.jsonl_path, "w", encoding="utf-8") as fh:
            for r in records:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    def _load_jsonl(self) -> list[dict]:
        if not os.path.exists(self.jsonl_path):
            return []
        records = []
        with open(self.jsonl_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    # --- Edge case: first run (no existing JSONL) ---
    def test_first_run_creates_records(self):
        self._write_ingested("v1.json", "syria", "همسات المساء", "[pause] هدوء المساء")
        new_records, stats, _ = builder.build_new_real_records(
            "syria", self.ingested_dir, existing_jsonl_path=self.jsonl_path,
        )
        self.assertEqual(stats["records_appended"], 1)
        self.assertEqual(stats["already_present"], 0)
        self.assertEqual(len(new_records), 1)

    # --- Edge case: re-run with no new data ---
    def test_rerun_no_new_data(self):
        self._write_ingested("v1.json", "syria", "همسات المساء", "[pause] هدوء المساء")
        # First run: creates record
        new1, stats1, _ = builder.build_new_real_records(
            "syria", self.ingested_dir, existing_jsonl_path=self.jsonl_path,
        )
        # Write to JSONL
        self._write_jsonl(new1)
        # Second run: no new data
        new2, stats2, _ = builder.build_new_real_records(
            "syria", self.ingested_dir, existing_jsonl_path=self.jsonl_path,
        )
        self.assertEqual(stats2["records_appended"], 0)
        self.assertEqual(stats2["already_present"], 1)
        self.assertEqual(len(new2), 0)

    # --- Edge case: add one new ingested file ---
    def test_add_new_ingested_appends_only_new(self):
        self._write_ingested("v1.json", "syria", "همسات المساء", "[pause] هدوء المساء")
        # First run
        new1, _, _ = builder.build_new_real_records(
            "syria", self.ingested_dir, existing_jsonl_path=self.jsonl_path,
        )
        self._write_jsonl(new1)
        # Add a second ingested file
        self._write_ingested("v2.json", "syria", "أصوات الطبيعة", "[pause] أصوات الطبيعة")
        # Second run
        new2, stats2, _ = builder.build_new_real_records(
            "syria", self.ingested_dir, existing_jsonl_path=self.jsonl_path,
        )
        self.assertEqual(stats2["records_appended"], 1)
        self.assertEqual(stats2["already_present"], 1)
        self.assertEqual(len(new2), 1)
        self.assertIn("أصوات الطبيعة", new2[0]["messages"][1]["content"])

    # --- Edge case: JSONL has existing synthetic records (not from ingested) ---
    def test_skips_synthetic_records_correctly(self):
        self._write_ingested("v1.json", "syria", "همسات المساء", "[pause] هدوء المساء")
        # Write a JSONL with 1 real + 1 synthetic record
        real = [builder._to_record("همسات المساء", "[pause] هدوء المساء", "syria")]
        synthetic = [builder._to_record("عنوان وهمي", "نص وهمي", "syria")]
        self._write_jsonl(real + synthetic)
        # Re-run should find 1 real already present, 0 new
        new, stats, existing_total = builder.build_new_real_records(
            "syria", self.ingested_dir, existing_jsonl_path=self.jsonl_path,
        )
        self.assertEqual(stats["already_present"], 1)
        self.assertEqual(stats["records_appended"], 0)
        self.assertEqual(len(new), 0)

    # --- Edge case: dialect mismatch filtered ---
    def test_dialect_mismatch_filtered(self):
        self._write_ingested("v1.json", "egypt", "عنوان مصري", "[pause] نص مصري")
        new, stats, _ = builder.build_new_real_records(
            "syria", self.ingested_dir, existing_jsonl_path=self.jsonl_path,
        )
        self.assertEqual(stats["dialect_mismatch"], 1)
        self.assertEqual(stats["records_appended"], 0)

    # --- Edge case: missing title or transcript ---
    def test_missing_title_or_transcript(self):
        self._write_ingested("v1.json", "syria", "", "[pause] نص بدون عنوان")
        self._write_ingested("v2.json", "syria", "عنوان بدون نص", "")
        new, stats, _ = builder.build_new_real_records(
            "syria", self.ingested_dir, existing_jsonl_path=self.jsonl_path,
        )
        self.assertEqual(stats["missing_title"], 1)
        self.assertEqual(stats["missing_transcript"], 1)
        self.assertEqual(stats["records_appended"], 0)

    # --- Edge case: empty ingestion directory ---
    def test_empty_ingestion_dir(self):
        with self.assertRaises(ValueError):
            builder.build_new_real_records(
                "syria", self.ingested_dir, existing_jsonl_path=self.jsonl_path,
            )


if __name__ == "__main__":
    unittest.main()