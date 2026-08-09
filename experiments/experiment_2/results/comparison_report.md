# Arabic ASMR Transcript Dialect Comparison — Results

Side experiment comparing three Ollama models on Arabic ASMR transcript quality under
three requested dialects (Standard / Syrian / Egyptian), at a fixed 5-minute duration.
Full method and inputs in `../README.md`. Transcripts kept in `../transcripts/` for
human review.

## Models

| Role | Model |
|---|---|
| Baseline (fine-tuned) | `arabic-asmr-syria:latest` |
| Candidate A | `mostafaasey25/nile-chat-12b:q6_k` |
| Candidate B | `command-r7b-arabic:latest` |

## Scoring rubric (each 1–5; overall = mean of 6)

1. **Storytelling richness** — descriptive detail, narrative arc, sensory language.
2. **Target dialect skillness** — how consistently the requested dialect is sustained.
3. **Relevance to user prompt** — stays on the girl/forest story.
4. **Personal attention heaviness** — how prominently the about-you (John, 38, engineer,
   Sara 8, Omar 5, reading, bedtime stories) is woven in.
5. **Timing adherence** — word count vs 425 (5 min × 85 wpm).
6. **Fish Audio pausing adherence** — valid `[pause]`/`[pause:Ns]`, no malformed
   `[pause: 3s]`, no `[break]`/`[long-break]`, no elongated words/meta-text.

## Per-dialect scores (human-assigned)

| Model | Dialect | Richness | Dialect | Relevance | Personal | Timing | Pauses | Overall |
|---|---|---|---|---|---|---|---|---|
| arabic-asmr-syria | standard | 3 | 3 | 5 | 5 | 2 | 5 | 3.83 |
| arabic-asmr-syria | syrian | 3 | 3 | 5 | 3 | 1 | 5 | 3.33 |
| arabic-asmr-syria | egyptian | 3 | 2 | 5 | 3 | 2 | 5 | 3.33 |
| nile-chat-12b | standard | 3 | 1 | 5 | 5 | 1 | 1 | 2.67 |
| nile-chat-12b | syrian | 4 | 1 | 5 | 5 | 1 | 2 | 3.00 |
| nile-chat-12b | egyptian | 4 | 5 | 5 | 5 | 1 | 1 | 3.50 |
| command-r7b-arabic | standard | 5 | 5 | 5 | 2 | 3 | 5 | 4.17 |
| command-r7b-arabic | syrian | 4 | 5 | 5 | 2 | 2 | 5 | 3.83 |
| command-r7b-arabic | egyptian | 5 | 5 | 5 | 5 | 4 | 5 | 4.83 |

## Per-dialect model ranking (which model to use for each dialect)

| Rank | Standard Arabic | Syrian | Egyptian |
|---|---|---|---|
| 🥇 1 | **command-r7b-arabic** (4.17) | **command-r7b-arabic** (3.83) | **command-r7b-arabic** (4.83) |
| 🥈 2 | arabic-asmr-syria (3.83) | arabic-asmr-syria (3.33) | nile-chat-12b (3.50) |
| 🥉 3 | nile-chat-12b (2.67) | nile-chat-12b (3.00) | arabic-asmr-syria (3.33) |

**`command-r7b-arabic` is the best model for all three dialects.**

## Model-level summary (mean across 3 dialects)

| Model | **Overall** |
|---|---|
| **command-r7b-arabic** | **4.28** |
| **arabic-asmr-syria (baseline)** | **3.50** |
| **nile-chat-12b** | **3.06** |

## Qualitative notes

- **command-r7b-arabic wins every dialect.** It is the only model that faithfully
  follows the requested dialect (MSA / Syrian / Egyptian) and emits valid, varied
  `[pause]` tags. Its Egyptian output is the single best transcript (4.83): correct
  dialect, strong personal attention (John as father, Sara & Omar as his children,
  bedtime-story framing), rich dialogue, and the best timing (339/425 words). Its
  weakness: for Standard and Syrian it **fails to weave in the about-you** (the story
  is a generic girl named Sara, not John's child).
- **Baseline (fine-tuned Syrian) is strong on personal attention and pauses** but
  degrades when asked for non-Syrian dialects: it leaks dialect (Egyptian `دي` in the
  Syrian run, Syrian `هاد` in the Egyptian run) and produces artifacts (Japanese `ジョン`,
  Chinese `约翰`, English `بتشتart` / `بيف applicable to`). It also under-lengths every
  run (186–219 words vs 425).
- **nile-chat-12b is weakest.** It produces **Egyptian even when asked for Standard or
  Syrian** (dialect skillness = 1 for both), emits **almost no pause tags** (0 in
  standard/egyptian), and adds meta-text (`أنا اسمي أم كلثوم` intro, `أتمنى تكون القصة
  عجبتك` outro). Its only strong point is personal attention.

## Recommendation

- **For all three dialects, prefer `command-r7b-arabic`** for ASMR transcript
  generation — it is the most dialect-faithful and pause-compliant.
- If **personal attention** is the top priority, the fine-tuned `arabic-asmr-syria`
  weaves the about-you more reliably, but at the cost of dialect fidelity and artifacts.
- `nile-chat-12b` is not recommended for dialect-controlled ASMR scripts.

> Note: this is a side experiment and does not change any application logic.
