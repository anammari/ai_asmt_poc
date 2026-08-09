# Arabic ASMR Transcript Model Comparison — Results

Side experiment comparing three Ollama models on Arabic ASMR transcript quality.
Full method and inputs in `../README.md`. Transcripts kept in `../transcripts/` for
human review.

## Models

| Role | Model | Size |
|---|---|---|
| Baseline (fine-tuned) | `arabic-asmr-syria:latest` | 4.7 GB |
| Candidate A | `mostafaasey25/nile-chat-12b:q6_k` | 9.7 GB |
| Candidate B | `command-r7b-arabic:latest` | 5.1 GB |

## Scoring rubric (1–5 each)

- **Relevance** — stays on the requested story (girl in a village discovering a forest
  of trees/flowers/butterflies) and weaves in the about-you (John, two children, bedtime stories).
- **Timing** — word count ≈ 85 words/min (85 / 255 / 425 for 1 / 3 / 5 min).
- **Pauses** — uses `[pause]`/`[pause:Ns]` with varied durations; no elongated words,
  markdown/meta-text, YouTube CTAs, or malformed tags.

**Overall** = mean of the three criteria (equal weight).

## Per-duration scores (human-assigned)

| Model | min | Timing | Relevance | Pauses | Overall |
|---|---|---|---|---|---|
| arabic-asmr-syria | 1 | 4 | 5 | 5 | 4.67 |
| arabic-asmr-syria | 3 | 2 | 5 | 3 | 3.33 |
| arabic-asmr-syria | 5 | 1 | 5 | 4 | 3.33 |
| nile-chat-12b | 1 | 2 | 5 | 2 | 3.00 |
| nile-chat-12b | 3 | 3 | 5 | 2 | 3.33 |
| nile-chat-12b | 5 | 1 | 3 | 2 | 2.00 |
| command-r7b-arabic | 1 | 1 | 5 | 5 | 3.67 |
| command-r7b-arabic | 3 | 3 | 5 | 3 | 3.67 |
| command-r7b-arabic | 5 | 2 | 5 | 3 | 3.33 |

## Model-level summary (mean across 1/3/5 min)

| Model | Timing | Relevance | Pauses | **Overall** |
|---|---|---|---|---|
| **arabic-asmr-syria (baseline)** | 2.33 | 5.0 | 4.0 | **3.78** |
| **command-r7b-arabic** | 2.00 | 5.0 | 3.67 | **3.56** |
| **nile-chat-12b** | 2.00 | 4.33 | 2.0 | **2.78** |

## Ranking

**1. `arabic-asmr-syria:latest` (3.78) > 2. `command-r7b-arabic:latest` (3.56) > 3. `nile-chat-12b:q6_k` (2.78)**

## Qualitative notes

- **Baseline (fine-tuned) wins** on timing and pause adherence. It is the only model
  that consistently emits varied `[pause]`/`[pause:Ns]` tags and weaves the about-you
  (John, two children, bedtime stories) into every duration. Weaknesses: under-length at
  3/5 min, a malformed `[.pause]` tag + `JOHN:` label at 3 min, and a Japanese katakana
  `ジョン` artifact at 5 min.
- **command-r7b-arabic is a close second** — best relevance (5.0) and good pause tags,
  but the worst timing (severely over-length at 1 min: 249 words vs 85). It also
  violates the no-meta rule at 3 min (ends with "هل تريد مني أن أواصل القصة؟") and emits
  malformed `[pause: 3s]` tags (space after colon) at 5 min that Fish Audio would not
  recognize.
- **nile-chat-12b is weakest** — emits **zero** pause tags at 3/5 min (uses `...`/`-`
  instead), drops the about-you entirely at 5 min, and is under-length at 1/5 min. Its
  dialect is Egyptian, not Syrian.

## Recommendation

Keep the fine-tuned `arabic-asmr-syria:latest` as `ARABIC_OLLAMA_MODEL`. Neither
general-purpose candidate beats it on the criteria that matter most for the app
(pause-tag adherence and consistent about-you personalization). `command-r7b-arabic`
is the only plausible alternative worth a second look, but its timing and meta-text
issues would need prompt-level fixes first.

> Note: this is a side experiment and does not change any application logic.
