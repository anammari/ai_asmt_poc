import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import importlib.util
spec = importlib.util.spec_from_file_location(
    "synth", REPO_ROOT / "finetuning" / "synthesize_syria.py"
)
synth = importlib.util.module_from_spec(spec)
spec.loader.exec_module(synth)


class TestSynthesizeEdgeCases(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        # Patch paths used by the module.
        self.orig_ingested = synth.INGESTED_DIR
        self.orig_jsonl = synth.JSONL_PATH
        self.orig_seeds = synth.SEEDS_PATH
        synth.INGESTED_DIR = os.path.join(self.tmpdir.name, "ingested")
        synth.JSONL_PATH = os.path.join(self.tmpdir.name, "training.jsonl")
        synth.SEEDS_PATH = os.path.join(self.tmpdir.name, "seeds.json")
        os.makedirs(synth.INGESTED_DIR)

        # Write a minimal seeds file.
        self._write_seeds([
            {"title": "جلسة استرخاء هادئة", "seed": "تنفس بعمق وراحه. استرخي.", "pool": "يوجا"},
            {"title": "أصوات المطر الهادئ", "seed": "استمع لصوت المطر. هدوء.", "pool": "بحر"},
        ])

    def tearDown(self):
        synth.INGESTED_DIR = self.orig_ingested
        synth.JSONL_PATH = self.orig_jsonl
        synth.SEEDS_PATH = self.orig_seeds
        self.tmpdir.cleanup()

    def _write_seeds(self, seeds: list[dict]):
        with open(synth.SEEDS_PATH, "w", encoding="utf-8") as fh:
            json.dump(seeds, fh, ensure_ascii=False)

    def _write_ingested(self, title: str, transcript: str, filename: str | None = None):
        if filename is None:
            filename = f"{title[:10]}.json"
        path = os.path.join(synth.INGESTED_DIR, filename)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({
                "dialect": "syria",
                "video_id": filename.replace(".json", ""),
                "title": title,
                "transcript": transcript,
                "ingested_at": "2025-01-01T00:00:00",
            }, fh, ensure_ascii=False)

    def _write_jsonl(self, records: list[dict]):
        with open(synth.JSONL_PATH, "w", encoding="utf-8") as fh:
            for r in records:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    def _load_jsonl(self) -> list[dict]:
        if not os.path.exists(synth.JSONL_PATH):
            return []
        records = []
        with open(synth.JSONL_PATH, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    # --- Edge case: first run, no existing JSONL ---
    def test_first_run_generates_synthetic(self):
        self._write_ingested("همسات المساء", "[pause] هدوء المساء")
        code = synth.main(["--count", "2"])
        self.assertEqual(code, 0)
        records = self._load_jsonl()
        self.assertEqual(len(records), 2)  # 2 synthetic

    # --- Edge case: re-run with same data generates nothing ---
    def test_rerun_no_new_synthetic(self):
        self._write_ingested("همسات المساء", "[pause] هدوء المساء")
        synth.main(["--count", "2"])
        code = synth.main(["--count", "2"])
        self.assertEqual(code, 0)
        records = self._load_jsonl()
        self.assertEqual(len(records), 2)  # Still 2, no duplicates

    # --- Edge case: increase --count after first run generates delta ---
    def test_increase_count_generates_delta(self):
        self._write_ingested("همسات المساء", "[pause] هدوء المساء")
        synth.main(["--count", "2"])
        synth.main(["--count", "4"])
        records = self._load_jsonl()
        self.assertEqual(len(records), 4)

    # --- Edge case: decrease --count is a no-op (never deletes) ---
    def test_decrease_count_is_noop(self):
        self._write_ingested("همسات المساء", "[pause] هدوء المساء")
        synth.main(["--count", "4"])
        synth.main(["--count", "2"])  # Lower target should be ignored
        records = self._load_jsonl()
        self.assertEqual(len(records), 4)  # Still 4

    # --- Edge case: force flag regenerates from scratch ---
    def test_force_removes_existing_synthetic(self):
        self._write_ingested("همسات المساء", "[pause] هدوء المساء")
        synth.main(["--count", "2"])
        synth.main(["--count", "2", "--force"])
        records = self._load_jsonl()
        self.assertEqual(len(records), 2)  # Regenerated, not appended

    # --- Edge case: deterministic output across runs ---
    def test_deterministic_output(self):
        self._write_ingested("همسات المساء", "[pause] هدوء المساء")
        synth.main(["--count", "3"])
        first = self._load_jsonl()
        # Reset and re-run
        os.remove(synth.JSONL_PATH)
        synth.main(["--count", "3"])
        second = self._load_jsonl()
        self.assertEqual(len(first), len(second))
        for i in range(len(first)):
            self.assertEqual(first[i]["messages"][2]["content"],
                             second[i]["messages"][2]["content"])

    # --- Edge case: adding new real data generates new synthetic ---
    def test_new_real_data_triggers_new_synthetic(self):
        self._write_ingested("همسات المساء", "[pause] هدوء المساء")
        synth.main(["--count", "2"])  # 2 synths based on 1 real
        # Add another real
        self._write_ingested("أصوات الطبيعة", "[pause] طبيعة", "v2.json")
        # Target = max(6, 2) = 6, so needs 4 more
        synth.main([])  # default count
        records = self._load_jsonl()
        self.assertGreater(len(records), 4)

    # --- Edge case: empty ingested directory ---
    def test_empty_ingested(self):
        code = synth.main(["--count", "2"])
        self.assertEqual(code, 1)  # Error exit

    # --- Edge case: existing JSONL has both real and synthetic; re-run preserves both ---
    def test_preserves_real_and_synthetic(self):
        self._write_ingested("همسات المساء", "[pause] هدوء المساء")
        # Manually write a JSONL with 1 real + 2 synthetic
        real = [synth._to_record("همسات المساء", "[pause] هدوء المساء")]
        s1 = synth._to_record("عنوان وهمي 1", "نص وهمي 1")
        s2 = synth._to_record("عنوان وهمي 2", "نص وهمي 2")
        self._write_jsonl(real + [s1, s2])
        # Re-run should detect 2 existing synthetic, target default = max(6,1) = 6, need 4 more
        synth.main([])
        records = self._load_jsonl()
        self.assertEqual(len(records), 1 + 2 + 4)  # 1 real + 2 old synth + 4 new synth


if __name__ == "__main__":
    unittest.main()