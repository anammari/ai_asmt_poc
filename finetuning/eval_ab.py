# finetuning/eval_ab.py
"""
A/B evaluation: base Ollama model vs the fine-tuned Arabic ASMR model.

Sends the same ASMR prompts (with the app's system prompt shape) to both
models through the local Ollama server and writes a side-by-side report
(finetuning/eval_report.md) with a scoring rubric.

Prereq: Ollama running locally with both models pulled/created, e.g.
  ollama pull ministral-3:8b
  ollama create arabic-asmr -f Modelfile   (after Unsloth GGUF export)

Usage:
  python finetuning/eval_ab.py --ft-model arabic-asmr:latest
  python finetuning/eval_ab.py --base-model ministral-3:8b --ft-model arabic-asmr:latest
"""
import argparse
import os
import sys

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from finetuning.build_dataset import SYSTEM_PROMPT  # noqa: E402

EVAL_PROMPTS = [
    "ساعدني على النوم بصوت نار مخيم هادئة",
    "صف لي جلسة استرخاء في مقهى مطر هادئ",
    "همسات تشجيع لطيفة بعد يوم عمل طويل",
]

RUBRIC = [
    "ASMR tone (intimate, soothing, whisper-paced)",
    "Pacing: [pause] markers used naturally and frequently",
    "Relevance to the requested topic",
    "Arabic fluency (MSA) and sensory imagery",
    "Cleanliness: no titles, markdown, or meta commentary",
]


def ask_ollama(base_url: str, model: str, topic: str, timeout: int = 120) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"User Instructions / Topic: {topic}"},
        ],
        "stream": False,
    }
    resp = requests.post(f"{base_url}/api/chat", json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Base vs fine-tuned Arabic ASMR A/B eval.")
    parser.add_argument("--base-model",
                        default=os.getenv("ARABIC_OLLAMA_MODEL", "arabic-asmr-syria:latest"))
    parser.add_argument("--ft-model", default="arabic-asmr:latest")
    parser.add_argument("--base-url",
                        default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    parser.add_argument("--prompts-file",
                        help="Optional text file with one Arabic topic per line.")
    parser.add_argument("--out",
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             "eval_report.md"))
    args = parser.parse_args(argv)

    if args.prompts_file:
        with open(args.prompts_file, encoding="utf-8") as fh:
            prompts = [line.strip() for line in fh if line.strip()]
    else:
        prompts = EVAL_PROMPTS

    lines = [
        "# Arabic ASMR Fine-tune A/B Evaluation",
        "",
        f"- Base model: `{args.base_model}`",
        f"- Fine-tuned: `{args.ft_model}`",
        f"- Ollama: {args.base_url}",
        "",
        "Score each dimension 1-5 per model after reading/listening:",
        "",
    ]
    for item in RUBRIC:
        lines.append(f"- [ ] {item}: base __ / ft __")
    lines.append("")

    had_error = False
    for i, topic in enumerate(prompts, start=1):
        print(f"[{i}/{len(prompts)}] {topic}")
        sections = {"base": "", "ft": ""}
        for label, model in (("base", args.base_model), ("ft", args.ft_model)):
            try:
                sections[label] = ask_ollama(args.base_url, model, topic)
            except Exception as exc:
                had_error = True
                sections[label] = f"_generation failed: {exc}_"
                print(f"  [warn] {label} model '{model}' failed: {exc}")

        lines += [
            f"## Prompt {i}: {topic}",
            "",
            f"### Base (`{args.base_model}`)",
            "",
            "```",
            sections["base"],
            "```",
            "",
            f"### Fine-tuned (`{args.ft_model}`)",
            "",
            "```",
            sections["ft"],
            "```",
            "",
        ]

    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"[done] A/B report -> {args.out}")
    return 1 if had_error else 0


if __name__ == "__main__":
    sys.exit(main())
