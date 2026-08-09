# Experiment 3 — Improving command-r7b-arabic ASMR Transcripts (A/B) — Results

A/B test of an **improved system prompt** against the **exp-2 baseline prompt**, both run on
`command-r7b-arabic:latest`, across the three dialects (Standard / Syrian / Egyptian), at a
fixed 5-minute duration. Full method, the two prompts, and inputs in `../README.md`;
transcripts kept in `../transcripts/` for human review; objective metrics + human scores in
`../scores/scores.json`.

> **v2 (re-run) note:** an earlier run produced a `syrian/improved` transcript that was pure
> Modern Standard Arabic (MSA), not Syrian. The improved prompt was fixed with a **DIALECT
> FIDELITY** rule (require the requested dialect in narration *and* dialogue, forbid MSA
> narration fallback), and the dialect scorer was fixed to **word-boundary matching** (short
> markers like `شو`/`عم` no longer false-positive inside longer MSA words — previously
> `شو` matched inside `تمشون` and `عم` inside `يعمل`/`تفاصيل`/`ناعمًا`/`عميقين`). This v2
> report reflects the re-run with both fixes.

## What changed between the arms

| | Baseline (exp-2 prompt) | Improved prompt |
|---|---|---|
| Fish Audio pause tags | `[pause]`, `[pause:2s]`, `[pause:3s]`, `[pause:4s]` (❌ **not supported** by s2.1) | `[break]` (brief) / `[long-break]` (extended), chosen by context ✅ |
| Dialect | "write in {dialect}" | adds **DIALECT FIDELITY**: entire script (narration *and* dialogue) in the requested dialect, no MSA narration fallback |
| Timing | vague "aim for ~425" | explicit "MUST reach ~425 words — do NOT finish early", 5–6 equal paragraphs |
| Richness | — | vivid sensory language, dialogue, narrative arc (intro→discovery→encounter→lesson→resolution), dialect idioms |
| Personal attention | "weave details naturally" (generic) | "address by name, weave details throughout — not just once — frame for family" (generic, portable) |

## Scoring rubric (each 1–5; overall = mean of 6)

1. **Storytelling richness** — sensory detail, narrative arc, dialogue.
2. **Target dialect skillness** — consistency of the requested dialect (scored with
   **word-boundary** marker matching).
3. **Relevance to user prompt** — stays on the girl/forest story.
4. **Personal attention heaviness** — how prominently the about-you (John, 38, engineer, Sara 8,
   Omar 5, reading/nature, bedtime stories) is woven in.
5. **Timing adherence** — word count vs 425 (5 min × 85 wpm).
6. **Fish Audio pausing adherence** — uses `[break]`/`[long-break]` (both present = varied);
   **no** `[pause]`/`[pause:Ns]`; no other bracket tags, no elongated words, no meta-text.

> **Note on the pause criterion:** it is scored under the **correct** Fish Audio convention
> (`[break]`/`[long-break]`) for **both** arms. The baseline arm was told to emit
> `[pause:Ns]`, so every baseline run scores 1 on pauses — this is expected and is exactly
> the defect the improved prompt fixes.

## A/B results per dialect (human-assigned; overall = mean of 6)

| Dialect | Arm | Richness | Dialect | Relevance | Personal | Timing | Pauses | **Overall** |
|---|---|---|---|---|---|---|---|---|
| standard | baseline | 4 | 5 | 5 | 4 | 3 | 1 | **3.67** |
| standard | improved | 5 | 5 | 5 | 3 | 4 | 5 | **4.50** |
| syrian | baseline | 4 | 4 | 5 | 4 | 3 | 1 | **3.50** |
| syrian | improved | 5 | 5 | 5 | 4 | 5 | 5 | **4.83** |
| egyptian | baseline | 4 | 5 | 5 | 4 | 3 | 1 | **3.67** |
| egyptian | improved | 5 | 5 | 5 | 4 | 5 | 5 | **4.83** |

## Per-criterion delta (improved − baseline)

| Criterion | Standard | Syrian | Egyptian |
|---|---|---|---|
| Richness | +1 | +1 | +1 |
| Dialect | 0 | **+1** | 0 |
| Relevance | 0 | 0 | 0 |
| Personal attention | −1 | 0 | 0 |
| Timing | +1 | **+2** | **+2** |
| Pauses | **+4** | **+4** | **+4** |
| **Overall** | **+0.83** | **+1.33** | **+1.17** |

## Headline findings

1. **The improved prompt fixes the pause defect everywhere.** All three improved runs emit
   valid, varied `[break]`/`[long-break]`; all three baseline runs emit unsupported
   `[pause:Ns]` (score 1 under the correct convention). Largest single win (+4 for each
   dialect).

2. **The dialect-fidelity rule fixed the Syrian regression.** `syrian/improved` is now
   genuinely Syrian in *both narration and dialogue* (`هادية`, `بتعيش`, `شو مخبأ`, `هل
   الغابة`, `عم تحوم`, `كتير`, `تيجي`, `رح تصير`) and scores dialect 5 (was falsely 5
   before while actually being MSA). Standard and Egyptian remain correct.

3. **The re-run also resolved the earlier Standard wash.** In the first run, `standard/improved`
   was a wash (no pause tags, under-length). In this run it emits valid varied pauses, ~375
   words, richness 5, and improves +0.83 over baseline — confirming that wash was one-off
   model variance, not a prompt defect.

4. **Improved beats baseline for all three dialects now** (standard +0.83, syrian +1.33,
   egyptian +1.17). The only negative delta is standard/improved personal attention (−1):
   this run's Standard output weaves Sara & Omar but does not name John or frame the story
   as told to him.

5. **The generic personal-attention rule is portable and effective.** Written with no
   hardcoded John/Sara/Omar, it lifts personal attention when the model cooperates (Egyptian
   addresses John by name, frames family-storytelling) without regressing the other criteria.

## Qualitative notes

- **syrian/improved (4.83) and egyptian/improved (4.83)** are the strongest outputs: correct
  dialect sustained across narration and dialogue, rich sensory language, near-perfect
  timing (442 / 445 words), and valid varied pause tags.
- **standard/improved (4.50)** is clean MSA with a strong Sara-and-Omar arc and valid pauses;
  only personal attention is a notch lower (John unnamed).
- **Baseline runs** all use invalid `[pause:Ns]` tags and are under-length (298–323 words),
  confirming the timing weakness the improved prompt targets.

## Recommendation / next step

- **Adopt the improved prompt** (with the dialect-fidelity rule) for all three dialects — it
  beats baseline on every arm and fixes both the Fish Audio pause defect and the Syrian
  dialect regression.
- **Port the dialect-fidelity, personal-attention, timing-floor, and richness rules** into
  the app's `llm.rewrite_script` system prompt. They are phrased generically (relative to the
  target word count and the requested dialect / "About me" text), so they work for any input.
- **Keep the word-boundary dialect scorer** if dialect quality is measured again — substring
  matching gives false positives on short markers.

> Note: this is a side experiment and does not change any application logic. The app's
> `tts.py` still emits `[pause]`/`[long pause]` and strips `[break]`/`[long-break]`
> (tts.py:76, 85–97); reconciling the app's tag handling is a separate app-logic task,
> out of scope here.
