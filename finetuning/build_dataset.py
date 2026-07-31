# finetuning/build_dataset.py
"""
Builds dialect-specific Arabic ASMR supervised fine-tuning (SFT) datasets.

Input: JSON files produced by ingest_YT_transcript.py in
       finetuning/data/ingested/<dialect>/

Output: Unsloth-compatible chat-format JSONL:
       finetuning/data/training/<dialect>/arabic_asmr_sft.jsonl

Append-only: re-running only adds new ingested records not already present.
Existing records (real or synthetic) are never modified or removed.

Usage:
  python finetuning/build_dataset.py --dialect syria
  python finetuning/build_dataset.py --dialect egypt
"""
import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

VALID_DIALECTS = {"syria", "egypt"}

DIALECT_DISPLAY = {
    "syria": "Syrian Arabic",
    "egypt": "Egyptian Arabic",
}

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


def _title_hash(title: str) -> str:
    return hashlib.sha256(title.strip().lower().encode("utf-8")).hexdigest()


def _user_topic(title: str, dialect: str) -> str:
    return f"User Instructions / Topic: [{DIALECT_DISPLAY[dialect]}] {title}"


def load_ingested_files(ingestion_dir: str) -> list[dict]:
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


def load_existing_records(jsonl_path: str) -> list[dict]:
    if not os.path.exists(jsonl_path):
        return []
    records = []
    with open(jsonl_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _to_record(title: str, transcript: str, dialect: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _user_topic(title, dialect)},
            {"role": "assistant", "content": transcript},
        ]
    }


def build_new_real_records(
    dialect: str,
    ingestion_dir: str,
    existing_jsonl_path: str | None = None,
) -> tuple[list[dict], dict, int]:
    """Return ONLY ingested records not already present in the existing JSONL."""
    if dialect not in VALID_DIALECTS:
        raise ValueError(
            f"Unsupported dialect '{dialect}'. Supported: {sorted(VALID_DIALECTS)}"
        )

    ingested = load_ingested_files(ingestion_dir)
    if not ingested:
        raise ValueError(f"No ingested .json files found in {ingestion_dir}")

    # Build hash set of existing user-message topics (covers both real & synthetic).
    existing_topics: set[str] = set()
    if existing_jsonl_path and os.path.exists(existing_jsonl_path):
        for record in load_existing_records(existing_jsonl_path):
            for msg in record.get("messages", []):
                if msg.get("role") == "user":
                    existing_topics.add(msg["content"].strip())
                    break

    new_records: list[dict] = []
    stats = {
        "ingested_files": len(ingested),
        "already_present": 0,
        "records_appended": 0,
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
        topic = _user_topic(title, dialect)
        if topic in existing_topics:
            stats["already_present"] += 1
            continue
        new_records.append(_to_record(title, transcript, dialect))

    stats["records_appended"] = len(new_records)
    return new_records, stats, len(existing_topics)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a dialect-specific Arabic ASMR fine-tuning dataset (append-only)."
    )
    parser.add_argument(
        "--dialect",
        choices=sorted(VALID_DIALECTS),
        required=True,
    )
    parser.add_argument("--ingestion-dir", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    ingestion_dir = args.ingestion_dir or _default_ingestion_dir(args.dialect)
    out_path = args.out or _default_output_path(args.dialect)

    new_records, stats, existing_total = build_new_real_records(
        args.dialect, ingestion_dir, existing_jsonl_path=out_path,
    )

    if not new_records:
        print(f"[stats] dialect={args.dialect}")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        print(f"[done] {stats['records_appended']} new records; JSONL already has {existing_total} records total")
        return 0

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as fh:
        for record in new_records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    new_total = existing_total + stats["records_appended"]
    print(f"[stats] dialect={args.dialect}")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print(f"[done] appended {stats['records_appended']} new records -> {out_path} (total: {new_total})")
    return 0


if __name__ == "__main__":
    sys.exit(main())