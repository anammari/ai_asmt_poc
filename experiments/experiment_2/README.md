# Experiment 2 — Arabic ASMR Dialect Comparison

A **side experiment** comparing how well three Ollama models follow a **requested Arabic
dialect** (Standard / Syrian / Egyptian) when generating ASMR transcripts, under a
stricter system prompt. Fully isolated in this folder; **does not modify any application
logic** — it calls `llm._call_ollama` read-only.

## Models under test

| Role | Model |
|---|---|
| Baseline (fine-tuned) | `arabic-asmr-syria:latest` |
| Candidate A | `mostafaasey25/nile-chat-12b:q6_k` |
| Candidate B | `command-r7b-arabic:latest` |

## Design

- **Duration fixed at 5 min** → target word count = 425 (85 words/min).
- **Vocal tone**: `Whispering`.
- **3 dialect iterations** of the system prompt, each sent to **all 3 models** →
  **9 transcripts** (3 models × 3 dialects).
- The system prompt is **identical across all models within an iteration**; only the
  dialect rule line changes between iterations.
- Because the prompt must vary by dialect, this experiment builds a **custom system
  prompt** and calls `llm._call_ollama(system_msg, user_msg, model=model)` directly
  (it cannot reuse `llm.rewrite_script`, which builds a fixed prompt).

### Dialect iterations

| Iteration | Dialect rule | Topic + About-you dialect |
|---|---|---|
| 1 | `Standard Arabic (no dialect)` | Standard Arabic (MSA) |
| 2 | `Syrian/Levantine Arabic dialect` | Syrian |
| 3 | `Egyptian Arabic dialect` | Egyptian |

### System prompt (per iteration, `{DIALECT_RULE}` substituted)

Adds strict emphasis on **Fish Audio pause compatibility**, **personal attention**, and
**relevance** on top of the app's base ASMR rules. Full template in
`inputs/test_inputs.json`.

### Test inputs (enriched, per dialect)

- **Topic**: girl in a quiet village discovers a hidden forest of tall trees, colorful
  flowers, butterflies, a clear river, and small animals, and learns a lesson about
  courage and friendship.
- **About-you**: John, 38, software engineer, two children (Sara 8, Omar 5), enjoys
  reading/nature/walking, reads bedtime stories. Written in the iteration's dialect.

## Scoring criteria (each 1–5; overall = mean of 6)

1. **Storytelling richness** — descriptive detail, narrative arc, sensory language.
2. **Target dialect skillness** — how consistently the requested dialect is sustained.
3. **Relevance to user prompt** — stays on the girl/forest story.
4. **Personal attention heaviness** — how prominently the about-you is woven in.
5. **Timing adherence** — word count vs 425.
6. **Fish Audio pausing adherence** — valid `[pause]`/`[pause:Ns]`, no malformed tags,
   no `[break]`/`[long-break]`, no elongated words/meta-text.

## How to run

```bash
uv run python experiments/experiment_2/run_dialect_comparison.py
```

Reads `OLLAMA_BASE_URL` from `.env` (default `http://localhost:11434`).

## Outputs

- `transcripts/<model>/<dialect>.txt` — the 9 generated transcripts (kept for human review).
- `scores/scores.json` — objective metrics + human-assigned 1–5 scores + per-dialect ranking.
- `results/comparison_report.md` — scoring table, per-dialect model ranking, overall scores, notes.

## Results

**`command-r7b-arabic` is the best model for all three dialects** (Standard 4.17,
Syrian 3.83, Egyptian 4.83). It is the only model that faithfully follows the requested
dialect and emits valid, varied pause tags. The fine-tuned baseline is strong on personal
attention but leaks dialect and produces artifacts when asked for non-Syrian dialects.
`nile-chat-12b` produces Egyptian even when asked for Standard/Syrian and emits almost
no pause tags.

Full scoring table and per-dialect ranking in
[`results/comparison_report.md`](results/comparison_report.md).
