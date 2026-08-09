"""Side experiment: compare Arabic ASMR transcript quality across three Ollama models.

Reuses the app's exact LLM call path (llm.rewrite_script) by overriding the
ARABIC_OLLAMA_MODEL env var per model. Does NOT modify any application logic.

Usage:
    uv run python experiments/run_arabic_model_comparison.py
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

# Allow importing the app's llm.py from the repo root.
# Script lives at experiments/experiment_1/, so the repo root is 3 levels up.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import llm  # noqa: E402

EXPERIMENTS_DIR = Path(__file__).resolve().parent
INPUTS_FILE = EXPERIMENTS_DIR / "inputs" / "test_inputs.json"
TRANSCRIPTS_DIR = EXPERIMENTS_DIR / "transcripts"
SCORES_FILE = EXPERIMENTS_DIR / "scores" / "scores.json"

# Story-element keywords from the test prompt.
STORY_KEYWORDS = ["بنت", "قرية", "غابة", "أشجار", "زهور", "فراشات"]
# About-you keywords (John, two children, bedtime stories).
ABOUT_KEYWORDS = ["جون", "طفلين", "قصص", "قبل النوم"]

# Elongated onomatopoeia the TTS would spell out letter-by-letter.
ELONGATED_RE = re.compile(r"(sh{2,}|ss{2,}|mm{2,}|zz{2,}|hh{2,})", re.IGNORECASE)
# YouTube-style CTA phrases the LLM sometimes hallucinates.
CTA_PHRASES = [
    "اشترك", "لايك", "subscribe", "like", "share", "comment",
    "تابعني", "تفضل بالاشتراك", "دعم القناة",
]
# Markdown / meta-text artifacts.
META_PATTERNS = [
    (re.compile(r"\*\*"), "bold-markdown"),
    (re.compile(r"\*"), "italic-markdown"),
    (re.compile(r"^#{1,6}\s", re.MULTILINE), "heading"),
    # Bracket groups that are NOT pause tags (pause tags are legitimate).
    (re.compile(r"\[(?!pause(?::\d+s)?\])[^\]]*\]", re.MULTILINE), "bracket-tag"),
]
PAUSE_RE = re.compile(r"\[pause(?::\d+s)?\]")
PAUSE_NS_RE = re.compile(r"\[pause:(\d+)s\]")


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
    # 1-5: 0-10% error -> 5, 10-20% -> 4, 20-35% -> 3, 35-50% -> 2, >50% -> 1
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
    elongated = ELONGATED_RE.findall(text)
    cta_hits = [p for p in CTA_PHRASES if p in text]
    meta_hits = [name for pat, name in META_PATTERNS if pat.search(text)]
    # 1-5: has pauses + varied durations + no artifacts -> 5; degrade per issue.
    score = 5
    if not pauses:
        score -= 3
    elif not varied:
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
        "elongated_words": elongated,
        "cta_hits": cta_hits,
        "meta_hits": meta_hits,
        "score": score,
    }


def score_relevance(text: str) -> dict:
    story_hits = [k for k in STORY_KEYWORDS if k in text]
    about_hits = [k for k in ABOUT_KEYWORDS if k in text]
    # 1-5: story on-topic + about-you woven in -> 5; degrade per missing element.
    score = 5
    if len(story_hits) < 3:
        score -= 1
    if len(story_hits) < 1:
        score -= 1
    if not about_hits:
        score -= 1
    score = max(1, min(5, score))
    return {
        "story_keywords_hit": story_hits,
        "about_keywords_hit": about_hits,
        "score": score,
    }


def generate_transcript(model: str, duration: int, inputs: dict, base_url: str) -> str:
    os.environ["ARABIC_OLLAMA_MODEL"] = model
    os.environ["OLLAMA_BASE_URL"] = base_url
    return llm.rewrite_script(
        context_text="",
        user_prompt=inputs["user_prompt"],
        provider="ollama",
        target_minutes=duration,
        vocal_tone=inputs["vocal_tone"],
        user_info=inputs["about_you"],
        language=inputs["language"],
    )


def main() -> None:
    inputs = load_inputs()
    base_url = load_env()
    wpm = inputs["words_per_minute"]
    models = inputs["models"]
    durations = inputs["durations_minutes"]

    results: dict = {"base_url": base_url, "inputs": inputs, "models": {}}

    for model in models:
        model_dir = TRANSCRIPTS_DIR / model.replace(":", "_").replace("/", "_")
        model_dir.mkdir(parents=True, exist_ok=True)
        results["models"][model] = {"durations": {}}
        for duration in durations:
            print(f"[generate] {model} @ {duration}min ...", flush=True)
            text = generate_transcript(model, duration, inputs, base_url)
            out_file = model_dir / f"{duration}min.txt"
            out_file.write_text(text, encoding="utf-8")

            timing = score_timing(text, duration, wpm)
            pauses = score_pauses(text)
            relevance = score_relevance(text)
            overall = round((timing["score"] + pauses["score"] + relevance["score"]) / 3, 2)

            results["models"][model]["durations"][str(duration)] = {
                "transcript_file": str(out_file),
                "timing": timing,
                "pauses": pauses,
                "relevance": relevance,
                "overall": overall,
            }

    SCORES_FILE.parent.mkdir(parents=True, exist_ok=True)
    SCORES_FILE.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    # Summary table.
    print("\n=== SUMMARY (objective metrics; human scoring pass still pending) ===")
    header = f"{'model':<38} {'min':<4} {'words':<6} {'target':<6} {'err%':<6} {'pauses':<7} {'rel':<4} {'overall':<7}"
    print(header)
    print("-" * len(header))
    for model in models:
        for duration in durations:
            d = results["models"][model]["durations"][str(duration)]
            print(
                f"{model:<38} {duration:<4} {d['timing']['word_count']:<6} "
                f"{d['timing']['target_words']:<6} {d['timing']['pct_error']:<6} "
                f"{d['pauses']['pause_count']:<7} {d['relevance']['score']:<4} {d['overall']:<7}"
            )
    print(f"\nTranscripts written under {TRANSCRIPTS_DIR}")
    print(f"Objective scores written to {SCORES_FILE}")


if __name__ == "__main__":
    main()
