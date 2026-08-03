import os
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from finetuning import train_unsloth_sft as trainer


class TestTrainArgs(unittest.TestCase):
    def test_required_dialect(self):
        with self.assertRaises(SystemExit):
            trainer.parse_args([])

    def test_default_dataset_and_output(self):
        args = trainer.parse_args(["--dialect", "syria"])
        self.assertEqual(args.dialect, "syria")
        self.assertTrue(args.resume)
        self.assertFalse(args.force_no_resume)
        self.assertTrue(
            args.dataset.endswith("finetuning/data/training/syria/arabic_asmr_sft.jsonl")
        )
        self.assertEqual(args.output_dir, "outputs/syria")
        self.assertEqual(args.ollama_name, "arabic-asmr-syria")

        args = trainer.parse_args(["--dialect", "egypt"])
        self.assertTrue(
            args.dataset.endswith("finetuning/data/training/egypt/arabic_asmr_sft.jsonl")
        )
        self.assertEqual(args.output_dir, "outputs/egypt")
        self.assertEqual(args.ollama_name, "arabic-asmr-egypt")

    def test_custom_paths(self):
        args = trainer.parse_args([
            "--dialect", "syria",
            "--dataset", "/tmp/data.jsonl",
            "--output-dir", "/tmp/outputs",
            "--ollama-name", "my-syrian-model",
            "--no-resume",
        ])
        self.assertEqual(args.dataset, "/tmp/data.jsonl")
        self.assertEqual(args.output_dir, "/tmp/outputs")
        self.assertEqual(args.ollama_name, "my-syrian-model")
        self.assertTrue(args.force_no_resume)

    def test_find_latest_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "checkpoint-100"))
            os.makedirs(os.path.join(tmp, "checkpoint-250"))
            os.makedirs(os.path.join(tmp, "adapter"))
            latest = trainer.find_latest_checkpoint(tmp)
            self.assertEqual(os.path.basename(latest), "checkpoint-250")
            self.assertTrue(os.path.isdir(latest))

    def test_no_checkpoint_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(trainer.find_latest_checkpoint(tmp))


if __name__ == "__main__":
    unittest.main()
