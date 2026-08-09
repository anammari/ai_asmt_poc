# Experiment 3 — Improving `command-r7b-arabic` ASMR Transcripts (A/B)

A **side experiment** testing whether an improved system prompt raises
`command-r7b-arabic`'s weak ASMR criteria (richness for Syrian, personal attention, timing,
and Fish Audio pause correctness) while keeping its strong criteria (dialect, relevance,
pause validity) at 5. Fully isolated in this folder; **does not modify any application
logic** — it calls `llm._call_ollama` read-only.

## Motivation

Experiment 2 showed `command-r7b-arabic` is the best model for all three dialects but has
weak spots: **personal attention** (2/5 for Standard & Syrian), **timing** (under-length:
310/235/339 vs 425 words), and **richness for Syrian** (4/5). It also exposed that the
exp-2 prompt's pause tags are **wrong for Fish Audio**: `[pause:2s]`/`[pause:3s]`/`[pause:4s]`
are **not supported** by the `s2.1` engine. The correct tags are `[break]` (brief pause) and
`[long-break]` (extended pause), hardcoded into the engine.

Experiment 3 corrects the pause convention and targets the weak criteria via prompt
engineering — with improvements phrased **generically** so they can later be ported into the
app's `llm.rewrite_script` for any user topic / "About me" text.

## Design

- **Model**: `command-r7b-arabic:latest` only.
- **Dialects**: Standard, Syrian, Egyptian (all 3).
- **A/B**: per dialect, run the **exp-2 prompt** (baseline arm) and the **improved prompt**
  (improved arm) → **6 transcripts** (3 dialects × 2 prompts).
- **Duration**: 5 min → target **425 words** (85 wpm). **Tone**: Whispering.
- **Inputs**: same enriched topic + about-you per dialect as exp 2 (in `inputs/test_inputs.json`).

## The two prompts

Both templates live in `inputs/test_inputs.json` as `baseline_prompt` (exp-2 verbatim,
including the `[pause:Ns]` rule) and `improved_prompt`. The improved prompt changes:

1. **Fish Audio pause tags** (fixes the wrong-tag issue): use ONLY `[break]` (brief) or
   `[long-break]` (extended), chosen by context; **never** `[pause]`, `[pause:2s]`,
   `[pause:3s]`, `[pause:4s]`, or other bracket tags.
2. **Dialect fidelity** (added in v2 to fix a regression): a DIALECT FIDELITY rule requires
   the **entire** script — narration *and* dialogue — in the requested dialect and forbids
   falling back to Standard/Formal Arabic narration. Generic, so it applies to whatever
   dialect is requested (trivially satisfied by Standard, reinforces Syrian/Egyptian).
3. **Timing adherence** (was 2–4): explicit "MUST reach the target word count (~425 words) —
   do NOT finish early; expand with sensory detail, dialogue, scene-setting" + "5–6
   paragraphs of roughly equal length". Phrased relative to target so it stays
   parameterizable (target = minutes × 85).
4. **Storytelling richness** (Syrian was 4): vivid sensory language (sights, sounds, smells,
   textures), dialogue, and a clear narrative arc (intro → discovery → encounter → lesson →
   resolution); use natural `{DIALECT_IDIOM}` idioms.
5. **Personal attention** (was 2 for Standard/Syrian): **generic** rule — "address the
   listener directly and by name; weave the provided personal details (name, age, family,
   work, hobbies) naturally and prominently throughout — not just once; if the info mentions
   family, frame the story as one told to them." No hardcoded John/Sara/Omar, so it ports to
   any "About me" text.
6. **Maintain the 5s** — dialect rule, relevance rule, and no-meta/no-elongated rules are
   unchanged.

> **Scoring fix (v2):** `score_dialect` uses **word-boundary** marker matching (a marker only
> counts when not flanked by an Arabic letter). Substring matching previously false-scored an
> MSA transcript as Syrian (e.g. `شو` inside `تمشون`, `عم` inside `يعمل`).

## Scoring criteria (each 1–5; overall = mean of 6)

Same 6 as exp 2, but the **pause criterion is redefined to the correct Fish Audio
convention** and applied to **both** arms:

1. **Storytelling richness** — sensory detail, narrative arc, dialogue.
2. **Target dialect skillness** — consistency of the requested dialect.
3. **Relevance to user prompt** — stays on the girl/forest story.
4. **Personal attention heaviness** — how prominently the about-you is woven in.
5. **Timing adherence** — word count vs 425.
6. **Fish Audio pausing adherence** — uses `[break]`/`[long-break]` (both present = varied);
   **no** `[pause]`/`[pause:Ns]`; no other bracket tags, no elongated words, no meta-text.

> Because the pause criterion is corrected, the baseline arm (told to use `[pause:Ns]`)
> scores low on pauses — expected, and it demonstrates the fix.

## How to run

```bash
uv run python experiments/experiment_3/run_command_r7b_improved.py
```

Reads `OLLAMA_BASE_URL` from `.env` (default `http://localhost:11434`). If your Ollama
models live in a custom store, set `OLLAMA_MODELS` (e.g. `OLLAMA_MODELS=~/ollama-models`).

## Outputs

- `transcripts/<dialect>/<baseline|improved>.txt` — the 6 generated transcripts.
- `scores/scores.json` — objective metrics **plus** human-assigned 1–5 scores per criterion.
- `results/comparison_report.md` — A/B table per dialect, per-criterion deltas, findings.

## Results (summary, v2 re-run)

| Dialect | Baseline | Improved | Δ |
|---|---|---|---|
| standard | 3.67 | 4.50 | **+0.83** |
| syrian | 3.50 | 4.83 | **+1.33** |
| egyptian | 3.67 | 4.83 | **+1.17** |

The improved prompt **fixes the pause defect** (all improved runs use valid
`[break]`/`[long-break]`) and **fixes the Syrian dialect regression** (the dialect-fidelity
rule produces genuine Syrian narration+dialogue). The earlier Standard wash did not
reproduce — this run's Standard improved emits valid pauses and beats baseline. Improved now
wins all three dialects. The only negative delta is standard/improved personal attention
(−1; John unnamed). Full analysis in
[`results/comparison_report.md`](results/comparison_report.md).
