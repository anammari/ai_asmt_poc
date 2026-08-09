# Experiments

Side experiments comparing Arabic ASMR transcript generation across three Ollama
models. Each experiment is fully isolated in its own folder and **does not modify any
application logic** — they reuse the app's `llm` module read-only.

## Models under test

| Role | Model |
|---|---|
| Baseline (fine-tuned) | `arabic-asmr-syria:latest` |
| Candidate A | `mostafaasey25/nile-chat-12b:q6_k` |
| Candidate B | `command-r7b-arabic:latest` |

## Experiments

- **[experiment_1/](experiment_1/README.md)** — Model comparison on Arabic ASMR
  transcript quality (relevance, timing, Fish Audio pause adherence) at 1/3/5 min.
  Result: baseline wins.
- **[experiment_2/](experiment_2/README.md)** — Dialect comparison: how well each model
  follows a requested Arabic dialect (Standard / Syrian / Egyptian) at 5 min, with a
  stricter system prompt (Fish Audio pause compatibility, personal attention, relevance).
  Includes a per-dialect model ranking.

## How to run

```bash
uv run python experiments/experiment_1/run_arabic_model_comparison.py
uv run python experiments/experiment_2/run_dialect_comparison.py
```

Both read `OLLAMA_BASE_URL` from `.env` (default `http://localhost:11434`).
