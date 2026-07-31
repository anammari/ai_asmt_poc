# finetuning/build_dataset.py
"""
Builds dialect-specific Arabic ASMR supervised fine-tuning (SFT) datasets.

Input: JSON files produced by ingest_YT_transcript.py in
       finetuning/data/ingested/<dialect>/

Output: Unsloth-compatible chat-format JSONL:
       finetuning/data/training/<dialect>/arabic_asmr_sft.jsonl

Each line (one sample) has:
  {"messages": [
      {"role": "system",    "content": <ASMR scriptwriter system prompt>},
      {"role": "user",      "content": "User Instructions / Topic: <video title>"},
      {"role": "assistant", "content": <Arabic ASMR transcript>},
  ]}

The system prompt mirrors the structure the app uses at inference time
(llm.rewrite_script with language="Arabic (العربية)"), so the fine-tuned
model learns the exact distribution the app will query.

Usage:
  python finetuning/build_dataset.py --dialect syria
  python finetuning/build_dataset.py --dialect egypt
"""
import argparse
import json
import os
import sys

# Allow `python finetuning/build_dataset.py` from the project root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

VALID_DIALECTS = {"syria", "egypt"}

# Dialect label for the user message (kept for traceability).
DIALECT_DISPLAY = {
    "syria": "Syrian Arabic",
    "egypt": "Egyptian Arabic",
}

# System prompt shape mirrors llm.rewrite_script's Arabic configuration.
SYSTEM_PROMPT = (
    "You are an expert ASMR scriptwriter generating scripts for an automated "
    "text-to-speech engine.\n"
    "Your goal is to write a script that takes exactly 2 minutes to speak "
    "slowly (strictly aim for ~170 words).\n"
    "The REQUIRED vocal tone is: WHISPERING.\n"
    "Language rule: Write the spoken script in Arabic (العربية) only.\n"
    "CRITICAL RULES:\n"
    "1. ONLY output the spoken script. NO titles, NO introductions, NO "
    "concluding remarks.\n"
    "2. NO meta-text, NO asterisks, NO markdown formatting. Do NOT use quotes.\n"
    "3. Insert [pause] or [pause:2s] (or up to [pause:4s]) frequently to "
    "dictate pacing.\n"
    "4. NEVER use elongated words. The TTS engine will spell them out "
    "letter-by-letter. Use standard words only.\n"
    "5. You MUST stay strictly relevant to the provided User Instructions."
)


def _default_output_path(dialect: str) -> str:
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "data",
        "training",
        dialect,
        "arabic_asmr_sft.jsonl",
    )


def _default_ingestion_dir(dialect: str) -> str:
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "data",
        "ingested",
        dialect,
    )


def load_ingested_files(ingestion_dir: str) -> list[dict]:
    """Read every .json file in the ingestion directory."""
    records: list[dict] = []
    if not os.path.isdir(ingestion_dir):
        raise FileNotFoundError(f"Ingestion directory not found: {ingestion_dir}")

    for filename in sorted(os.listdir(ingestion_dir)):
        if not filename.endswith(".json"):
            continue
        path = os.path.join(ingestion_dir, filename)
        with open(path, encoding="utf-8") as fh:
            records.append(json.load(fh))
    return records


def _to_record(title: str, transcript: str, dialect: str) -> dict:
    topic = f"[{DIALECT_DISPLAY[dialect]}] {title}"
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"User Instructions / Topic: {topic}"},
            {"role": "assistant", "content": transcript},
        ]
    }


def build_dataset(dialect: str, ingestion_dir: str) -> tuple[list[dict], dict]:
    """Construct chat-format JSONL records from ingested YouTube metadata."""
    if dialect not in VALID_DIALECTS:
        raise ValueError(
            f"Unsupported dialect '{dialect}'. Supported: {sorted(VALID_DIALECTS)}"
        )

    ingested = load_ingested_files(ingestion_dir)
    if not ingested:
        raise ValueError(f"No ingested .json files found in {ingestion_dir}")

    records: list[dict] = []
    stats = {
        "ingested_files": len(ingested),
        "records_produced": 0,
        "missing_title": 0,
        "missing_transcript": 0,
        "dialect_mismatch": 0,
    }

    for item in ingested:
        item_dialect = (item.get("dialect") or "").strip().lower()
        if item_dialect and item_dialect != dialect:
            stats["dialect_mismatch"] += 1
            continue
        title = (item.get("title") or "").strip()
        transcript = (item.get("transcript") or "").strip()
        if not title:
            stats["missing_title"] += 1
            continue
        if not transcript:
            stats["missing_transcript"] += 1
            continue
        records.append(_to_record(title, transcript, dialect))

    stats["records_produced"] = len(records)
    return records, stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a dialect-specific Arabic ASMR fine-tuning dataset."
    )
    parser.add_argument(
        "--dialect",
        choices=sorted(VALID_DIALECTS),
        required=True,
        help="Target dialect (syria or egypt).",
    )
    parser.add_argument(
        "--ingestion-dir",
        help="Directory containing ingested JSON files. Auto-derived from --dialect if omitted.",
    )
    parser.add_argument(
        "--out",
        help="Output JSONL path. Auto-derived from --dialect if omitted.",
    )
    args = parser.parse_args(argv)

    ingestion_dir = args.ingestion_dir or _default_ingestion_dir(args.dialect)
    out_path = args.out or _default_output_path(args.dialect)

    records, stats = build_dataset(args.dialect, ingestion_dir)
    if not records:
        print("error: no valid samples produced.", file=sys.stderr)
        return 1

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[stats] dialect={args.dialect}")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print(f"[done] wrote {len(records)} samples -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
