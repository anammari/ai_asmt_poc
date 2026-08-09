"""Side experiment: compare Arabic ASMR transcript quality across three Ollama models
under three requested Arabic dialects (Standard / Syrian / Egyptian).

Builds a custom system prompt (dialect rule + strict Fish Audio pause / personal
attention / relevance emphasis) and calls llm._call_ollama directly. Does NOT modify
any application logic.

Usage:
    uv run python experiments/experiment_2/run_dialect_comparison.py
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

# Allow importing the app's llm.py from the repo root.
# Script lives at experiments/experiment_2/, so the repo root is 3 levels up.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import llm  # noqa: E402

EXPERIMENTS_DIR = Path(__file__).resolve().parent
INPUTS_FILE = EXPERIMENTS_DIR / "inputs" / "test_inputs.json"
TRANSCRIPTS_DIR = EXPERIMENTS_DIR / "transcripts"
SCORES_FILE = EXPERIMENTS_DIR / "scores" / "scores.json"

# Story-element keywords from the enriched topic.
STORY_KEYWORDS = ["بنت", "فتاة", "قرية", "غابة", "أشجار", "زهور", "فراشات", "نهر", "حيوانات"]
# About-you keywords (John, 38, Sara, Omar, engineer, reading, nature, bedtime stories).
ABOUT_KEYWORDS = ["جون", "سارة", "عمر", "38", "مهندس", "برمجيات", "قراءة", "طبيعة", "قصص", "قبل النوم", "طفلين"]

# Dialect marker keywords (hints for the human scoring pass).
DIALECT_MARKERS = {
    "standard": ["التي", "الذي", "كانت", "لقد", "إنّ", "أيضاً"],
    "syrian": ["كتير", "شو", "هيك", "عم", "إلهن", "بدي", "ليش", "منيح", "هلق", "حلوة"],
    "egyptian": ["إزاي", "أوي", "كده", "عايز", "بتاع", "ليه", "إيه", "دلوقتي", "خالص", "محدش"],
}

# Elongated onomatopoeia the TTS would spell out letter-by-letter.
ELONGATED_RE = re.compile(r"(sh{2,}|ss{2,}|mm{2,}|zz{2,}|hh{2,})", re.IGNORECASE)
# YouTube-style CTA phrases the LLM sometimes hallucinates.
CTA_PHRASES = [
    "اشترك", "لايك", "subscribe", "like", "share", "comment",
    "تابعني", "تفضل بالاشتراك", "دعم القناة",
]
# Meta-text / markdown artifacts (excluding legitimate pause tags).
META_PATTERNS = [
    (re.compile(r"\*\*"), "bold-markdown"),
    (re.compile(r"\*"), "italic-markdown"),
    (re.compile(r"^#{1,6}\s", re.MULTILINE), "heading"),
    (re.compile(r"\[(?!pause(?::\d+s)?\])[^\]]*\]", re.MULTILINE), "bracket-tag"),
]
PAUSE_RE = re.compile(r"\[pause(?::\d+s)?\]")
PAUSE_NS_RE = re.compile(r"\[pause:(\d+)s\]")
# Malformed pause tags (space after colon) that Fish Audio would not recognize.
MALFORMED_PAUSE_RE = re.compile(r"\[pause:\s+\d+s\]")
# Non-Fish-Audio pause tags the prompt forbids.
FORBIDDEN_TAGS = ["[break]", "[long-break]"]


def load_env(base_url_default: str = "http://localhost:11434") -> str:
    """Read OLLAMA_BASE_URL from .env (manual parse; no python-dotenv dep)."""
    env_file = REPO_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("OLLAMA_BASE_URL="):
                return line.split("=", 1)[1].strip() or base_url_default
    return base_url_default


def load_inputs() -> dict:
    return json.loads(INPUTS_FILE.read_text(encoding="utf-8"))


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def score_timing(text: str, target_minutes: int, wpm: int) -> dict:
    target = target_minutes * wpm
    count = word_count(text)
    pct_error = abs(count - target) / target * 100.0
    if pct_error <= 10:
        score = 5
    elif pct_error <= 20:
        score = 4
    elif pct_error <= 35:
        score = 3
    elif pct_error <= 50:
        score = 2
    else:
        score = 1
    return {"target_words": target, "word_count": count, "pct_error": round(pct_error, 1), "score": score}


def score_pauses(text: str) -> dict:
    pauses = PAUSE_RE.findall(text)
    ns_pauses = PAUSE_NS_RE.findall(text)
    varied = len(set(ns_pauses)) >= 2 if ns_pauses else False
    malformed = MALFORMED_PAUSE_RE.findall(text)
    forbidden = [t for t in FORBIDDEN_TAGS if t in text]
    elongated = ELONGATED_RE.findall(text)
    cta_hits = [p for p in CTA_PHRASES if p in text]
    meta_hits = [name for pat, name in META_PATTERNS if pat.search(text)]
    score = 5
    if not pauses:
        score -= 3
    elif not varied:
        score -= 1
    if malformed:
        score -= 1
    if forbidden:
        score -= 1
    if elongated:
        score -= 1
    if cta_hits:
        score -= 1
    if meta_hits:
        score -= 1
    score = max(1, min(5, score))
    return {
        "pause_count": len(pauses),
        "varied_durations": varied,
        "malformed_pause_tags": malformed,
        "forbidden_tags": forbidden,
        "elongated_words": elongated,
        "cta_hits": cta_hits,
        "meta_hits": meta_hits,
        "score": score,
    }


def score_relevance(text: str) -> dict:
    story_hits = [k for k in STORY_KEYWORDS if k in text]
    score = 5
    if len(story_hits) < 3:
        score -= 1
    if len(story_hits) < 1:
        score -= 1
    score = max(1, min(5, score))
    return {"story_keywords_hit": story_hits, "score": score}


def score_personal_attention(text: str) -> dict:
    about_hits = [k for k in ABOUT_KEYWORDS if k in text]
    score = 5
    if len(about_hits) < 3:
        score -= 1
    if len(about_hits) < 1:
        score -= 1
    score = max(1, min(5, score))
    return {"about_keywords_hit": about_hits, "score": score}


def score_dialect(text: str, dialect: str) -> dict:
    markers = DIALECT_MARKERS.get(dialect, [])
    hits = [k for k in markers if k in text]
    score = 5
    if not hits:
        score -= 2
    elif len(hits) < 2:
        score -= 1
    score = max(1, min(5, score))
    return {"markers_hit": hits, "score": score}


def build_system_prompt(template: str, dialect_rule: str) -> str:
    return template.replace("{DIALECT_RULE}", dialect_rule)


def build_user_message(topic: str, about_you: str) -> str:
    return (
        f"User Instructions / Topic: {topic}\n\n"
        f"User's Personal Info for Personal Attention:\n{about_you}"
    )


def main() -> None:
    inputs = load_inputs()
    base_url = load_env()
    wpm = inputs["words_per_minute"]
    duration = inputs["duration_minutes"]
    models = inputs["models"]
    dialects = inputs["dialects"]
    template = inputs["system_prompt_template"]

    results: dict = {"base_url": base_url, "inputs": inputs, "models": {}}

    for model in models:
        model_dir = TRANSCRIPTS_DIR / model.replace(":", "_").replace("/", "_")
        model_dir.mkdir(parents=True, exist_ok=True)
        results["models"][model] = {"dialects": {}}
        for dialect_name, dialect in dialects.items():
            print(f"[generate] {model} @ {dialect_name} ...", flush=True)
            system_msg = build_system_prompt(template, dialect["rule"])
            user_msg = build_user_message(dialect["topic"], dialect["about_you"])
            text = llm._call_ollama(system_msg, user_msg, model=model)
            out_file = model_dir / f"{dialect_name}.txt"
            out_file.write_text(text, encoding="utf-8")

            timing = score_timing(text, duration, wpm)
            pauses = score_pauses(text)
            relevance = score_relevance(text)
            personal = score_personal_attention(text)
            dialect_score = score_dialect(text, dialect_name)
            richness = {"word_count": timing["word_count"], "score": min(5, max(1, timing["score"] + 1))}
            overall = round(
                (richness["score"] + dialect_score["score"] + relevance["score"]
                 + personal["score"] + timing["score"] + pauses["score"]) / 6, 2
            )

            results["models"][model]["dialects"][dialect_name] = {
                "transcript_file": str(out_file),
                "richness": richness,
                "dialect": dialect_score,
                "relevance": relevance,
                "personal_attention": personal,
                "timing": timing,
                "pauses": pauses,
                "overall": overall,
            }

    SCORES_FILE.parent.mkdir(parents=True, exist_ok=True)
    SCORES_FILE.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== SUMMARY (objective metrics; human scoring pass still pending) ===")
    header = f"{'model':<38} {'dialect':<9} {'words':<6} {'err%':<6} {'pauses':<7} {'dialect':<8} {'rel':<4} {'pers':<5} {'overall':<7}"
    print(header)
    print("-" * len(header))
    for model in models:
        for dialect_name in dialects:
            d = results["models"][model]["dialects"][dialect_name]
            print(
                f"{model:<38} {dialect_name:<9} {d['timing']['word_count']:<6} "
                f"{d['timing']['pct_error']:<6} {d['pauses']['pause_count']:<7} "
                f"{d['dialect']['score']:<8} {d['relevance']['score']:<4} "
                f"{d['personal_attention']['score']:<5} {d['overall']:<7}"
            )
    print(f"\nTranscripts written under {TRANSCRIPTS_DIR}")
    print(f"Objective scores written to {SCORES_FILE}")


if __name__ == "__main__":
    main()
