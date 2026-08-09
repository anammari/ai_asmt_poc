# Arabic ASMR Transcript Model Comparison (Side Experiment)

A **side experiment** comparing the Arabic ASMR transcript quality of three Ollama
models. It is fully isolated in this `experiments/` folder and **does not modify any
application logic** (`app.py`, `llm.py`, `tts.py`, etc.). It reuses the app's exact
LLM call path (`llm.rewrite_script`) read-only, so transcripts are generated exactly
as the app would produce them.

## Models under test

| Role | Model | Notes |
|---|---|---|
| Baseline (fine-tuned) | `arabic-asmr-syria:latest` | The app's current `ARABIC_OLLAMA_MODEL` |
| Candidate A | `mostafaasey25/nile-chat-12b:q6_k` | General-purpose Arabic chat (12B) |
| Candidate B | `command-r7b-arabic:latest` | General-purpose Arabic chat (7B) |

## Test inputs (identical for all three models)

Defined in `inputs/test_inputs.json`:

- **User prompt** (from README "Test Case: Arabic Storytelling ASMR"):
  `احكيلي قصة عن بنت بتعيش بقرية صغيرة وتكتشف غابة جديدة مليانة أشجار وزهور والفراشات عم تحوم حواليها`
- **About you** (Syrian dialect):
  `اسمي جون. عندي طفلين وكتير بقرا إلهن قصص قبل النوم`
- **Vocal tone**: `Whispering` · **Language**: `Arabic`
- **Target durations**: `1`, `3`, `5` minutes → **9 transcripts** (3 models × 3 durations)

## Assessment criteria (each scored 1–5)

1. **Relevance to prompt / user request / about-user** — stays on the requested story
   (girl in a village discovering a forest of trees, flowers, butterflies) and weaves
   in the about-you details (John, two children, bedtime stories).
2. **Timing adherence** — word count ≈ target (85 words/min → 85 / 255 / 425 words for
   1 / 3 / 5 min). Score by closeness to target.
3. **Fish Audio pause adherence** — uses `[pause]` / `[pause:Ns]` tags with varied
   durations; no elongated words (`shhhh`, `sssss`), no markdown/meta-text, no YouTube
   CTAs, no titles/intro/conclusion.

**Overall score** per model = mean of the three criterion scores (equal weight).

## How to run

```bash
# 1. Ensure the candidate models are pulled into the custom store:
OLLAMA_MODELS=~/ollama-models ollama pull mostafaasey25/nile-chat-12b:q6_k
OLLAMA_MODELS=~/ollama-models ollama pull command-r7b-arabic:latest

# 2. Run the experiment (generates transcripts + objective scores):
uv run python experiments/run_arabic_model_comparison.py
```

The script reads `OLLAMA_BASE_URL` from `.env` (default `http://localhost:11434`),
overrides `ARABIC_OLLAMA_MODEL` per model, and calls `llm.rewrite_script(...)`.

## Outputs

- `transcripts/<model>/<N>min.txt` — the 9 generated transcripts (kept for human review).
- `scores/scores.json` — objective metrics (word count, pause tags, keyword hits) plus
  the human-assigned 1–5 scores.
- `results/comparison_report.md` — the scoring table, overall scores, and comparison notes.

## Results

**Ranking: `arabic-asmr-syria:latest` (3.78) > `command-r7b-arabic:latest` (3.56) > `nile-chat-12b:q6_k` (2.78).**

The fine-tuned baseline wins on timing and pause-tag adherence and is the only model
that consistently weaves in the about-you. Neither general-purpose candidate beats it.
Full scoring table, qualitative notes, and recommendation in
[`results/comparison_report.md`](results/comparison_report.md).
