# CLAUDE.md

This file gives Claude Code (and any other AI coding agent) the context needed to work effectively in this repository. Read it fully before making changes.

## 1. Project Overview

**AI ASMR Proof of Concept** ("Zenith ASMR PoC" internally) is a Streamlit application that turns a topic/URL/voice prompt into a spoken ASMR (Autonomous Sensory Meridian Response) audio track. It is a **cascaded STT → LLM → TTS pipeline**:

1. **Input** — the user provides a topic via typed text (Arabic) or a recorded voice prompt (English), optionally with a source URL for context and personal details for a "personalized" script.
2. **STT** — English voice prompts are transcribed locally with Whisper (`transformers` pipeline).
3. **LLM rewrite** — an LLM (OpenRouter for English, Gemini/Ollama for Arabic) rewrites the raw topic/context into a slow-paced, whisper-style ASMR script with `[pause]` markers.
4. **Sanitization** — the raw script is cleaned and pause tags are converted into `<<SILENCE<n>MS>>` markers consumed by the TTS layer.
5. **TTS synthesis** — **both English and Arabic** use the **Fish Audio `s2.1-pro-free`** cloud API with inline S2 direction tags (`[whispering]`, `[soft]`, `[break]`).
6. **Output** — a `.wav` file is played back in-app and downloadable; a full session log is written per run.

There is also a **secondary sub-project**, `finetuning/`, which fine-tunes a dialect-specific (Syrian / Egyptian) Arabic ASMR scriptwriting LLM via Unsloth QLoRA, served back to the app through Ollama.

### Objectives
- Zero/low marginal cost: prefer local, open-source, or free-tier models over paid APIs.
- High-quality whisper/soft-spoken delivery in **both English and Arabic**.
- Personalization: weave user-provided personal details naturally into scripts.
- Arabic is a first-class, actively-developed track (dialect fine-tuning, dedicated TTS backend, Arabic text normalization).

## 2. Tech Stack

| Layer | Technology |
|---|---|
| Language / runtime | Python 3.12 (`.python-version`), managed with **`uv`** (not pip/poetry) |
| UI | Streamlit (`st.audio_input`, `st.status`, custom CSS injection for dark-friendly inputs) |
| STT | `transformers` pipeline, `openai/whisper-small` (configurable via `STT_MODEL`), `insanely-fast-whisper` dependency present for MPS/Apple Silicon speed |
| LLM (English) | **OpenRouter** (`OPENROUTER_MODEL`, default `google/gemma-4-26b-a4b-it:free`, OpenAI-compatible REST via `requests`) with fallback to local Ollama, then a local fallback script |
| LLM (Arabic) | Google Gemini (`GEMINI_MODEL`, HTTP REST calls, no SDK) with fallback model list, transient-error retry/backoff, and fallback to local Ollama |
| LLM (local / fine-tuned) | Ollama (`OLLAMA_BASE_URL`, `OLLAMA_MODEL`, e.g. `ministral-3:8b` or a fine-tuned `arabic-asmr-<dialect>` model) |
| LLM (not implemented) | OpenAI provider is stubbed (`_call_openai` raises `NotImplementedError`) |
| TTS (English + Arabic) | **Fish Audio `s2.1-pro-free`** REST API (`fish_audio_tts.py`) — the *sole* TTS backend for both languages (Kokoro-82M and a previous SILMA/F5-TTS approach were removed) |
| Content fetching | `requests` + `BeautifulSoup` for generic URLs, `langchain_community.WikipediaLoader` for Wikipedia/free-text queries |
| Arabic NLP | Custom `preprocess_arabic.py` (orthographic normalization, optional diacritization via `mishkal`/`camel_tools`) |
| Fine-tuning | Unsloth QLoRA (Colab/Linux CUDA only — **not** macOS), `trl`, `datasets`, GGUF export for Ollama |
| Containerization | Docker (`python:3.12-slim`, `ffmpeg` system dep) |
| Testing | `pytest`, smoke tests + regression tests, mostly using `unittest.mock` |

## 3. Repository Structure

```
ai_asmr_poc/
├── app.py                    # Streamlit entrypoint + orchestration + sanitize_text()
├── content_fetcher.py        # URL/Wikipedia/movie/lyrics context fetching with fallbacks
├── llm.py                    # OpenRouter + Gemini + Ollama script rewriting, retries, key validation
├── stt.py                    # Whisper STT pipeline + transcript verification helpers
├── tts.py                    # Fish Audio synthesis, silence-marker → audio conversion
├── fish_audio_tts.py         # Standalone Fish Audio API client + CLI eval harness
├── preprocess_arabic.py      # Arabic normalization/diacritization (also has a CLI)
├── Dockerfile                # Container build
├── pyproject.toml / uv.lock  # Dependencies (managed via `uv`, NOT pip directly)
├── .env.example              # Template for required environment variables
├── logs/                     # Per-session generation logs (asmr_session_<id>.log) — gitignored
├── output/                   # Generated audio / Fish Audio assessment reports — gitignored
├── input/                    # Sample/manual test audio+text fixtures
├── tests/                    # pytest smoke + regression tests (see §7)
├── finetuning/                # Arabic dialect fine-tuning sub-pipeline (see §6)
├── prompts/                  # Planning docs for Arabic finetuning (arabic-finetuning-*.md)
├── qa_test_issues.md         # ACTIVE bug list / improvement requests — gitignored but authoritative (see §8)
├── feasibility_study_gemini.md, poc_guide.md, fine-tune-resources.md  # Background research / design docs
└── .github/copilot-instructions.md  # Legacy Copilot guidance (superseded in spirit by this file; kept for reference)
```

Note: `.gitignore` excludes **all Markdown files except `README.md` and `finetuning/README.md`**, so most `.md` docs in this repo (including this file, if committed, and `qa_test_issues.md`) are local-only / not tracked by git unless explicitly added.

## 4. Environment & Setup

Copy `.env.example` → `.env` before running. Key variables:

- `ENGLISH_LLM_PROVIDER` = `openrouter` | `gemini` | `openai` (unimplemented) | `ollama`
- `ARABIC_LLM_PROVIDER` = `gemini` | `ollama`
- `OPENROUTER_API_KEY`, `OPENROUTER_MODEL` (default `google/gemma-4-26b-a4b-it:free`)
- `GEMINI_API_KEY`, `GEMINI_MODEL` (default `gemini-2.5-flash-lite`), `GEMINI_FALLBACK_MODELS`
- `OLLAMA_BASE_URL` (use `http://host.docker.internal:11434` when app runs in Docker but Ollama runs natively on the host), `OLLAMA_MODEL`
- `OUTPUT_DIR` (default `./output`)
- `FISH_AUDIO_API_KEY`, `FISH_AUDIO_MODEL` (default `s2.1-pro-free`), `FISH_AUDIO_VOICE_ID`
- `STT_MODEL` (default `openai/whisper-small`)
- `AR_DIACRITIZER` = `mishkal` (default) | `camel` | `none`

### Native run (macOS/Linux)
```bash
uv sync                           # install all deps from pyproject.toml/uv.lock
uv run python app.py              # self-bootstraps into `streamlit run` if invoked directly
```

### Docker run
```bash
docker build -t ai-asmr-poc .
docker run --rm --env-file .env -v $(pwd)/output:/app/output -p 8000:8000 ai-asmr-poc
```

**Always use `uv` for dependency management** (`uv add <pkg>`, `uv sync`), never bare `pip install`.

## 5. Core Pipeline Details (read before editing `app.py`, `llm.py`, or `tts.py`)

- **`sanitize_text()` in `app.py`** is the single place where LLM output is turned into TTS-ready text: it converts newlines/punctuation into `<<SILENCE<n>MS>>` markers, expands `[pause]`/`[pause:Ns]` tags (clamped to 200–5000ms), strips leftover bracket tags and markdown artifacts, strips elongated onomatopoeia (`shhhh`, `sssss`, etc. — these make TTS spell letters out loud), strips YouTube-style CTA phrases the LLM sometimes hallucinates from training data, and finally calls `normalize_arabic()`.
- **Language routing**: Arabic scripts are *always* generated via Gemini (`llm.rewrite_script` forces `provider = "gemini"` when `is_arabic`), regardless of `LLM_PROVIDER` — this is intentional per current logic, but **see the open QA item in §8 requesting this be changed** to use `GEMINI_MODEL` specifically as the Arabic default regardless of provider.
- **Voice routing in `tts.synthesize()`**: paragraphs (split on blank lines) alternate across selected voices. **All** voices are Fish Audio (`fish_<reference_id>` IDs) and are routed to `generate_fish_audio()`. Silence markers (`<<SILENCE<n>MS>>`) are converted into Fish Audio `[pause]`/`[long pause]` tag combinations (Fish Audio has no millisecond-precise silence control).
- **`llm.rewrite_script()`** builds a system prompt with strict formatting rules (no titles/markdown/quotes, `[pause]` tags required, no elongated words) and language/tone-specific additions (Arabic gets an explicit 4–6 paragraph / varied-pause-duration rule). English uses OpenRouter (`_call_openrouter`) with fallback to Ollama then a local fallback script; Arabic uses Gemini, which retries transient errors (429/500/503) with exponential backoff across a deduplicated candidate model list, then falls back to Ollama, then to a trivial local fallback script.
- **Voice catalogs** are hardcoded in `app.py`:
  - English: `ENGLISH_VOICE_CATEGORIES` (Fish Audio `fish_<reference_id>` IDs), split into "Soft Spoken" / "Whispering".
  - Arabic: `ARABIC_VOICES` (Fish Audio `fish_<reference_id>` IDs), also split by tone. Display names come from `fish-audio-models.txt`.

## 6. Arabic Fine-Tuning Sub-Pipeline (`finetuning/`)

Goal: produce dialect-specific (Syrian, Egyptian) Arabic ASMR scriptwriter models trained on **real YouTube ASMR transcripts** (the old Gemini-synthetic-data approach was removed).

Pipeline: `metadata.csv` (Dialect,URL) → `ingest_YT_transcript.py` (yt-dlp + youtube-transcript-api, writes `data/ingested/<dialect>/*.json`) → `build_dataset.py --dialect <syria|egypt>` (writes Unsloth chat-format JSONL to `data/training/<dialect>/arabic_asmr_sft.jsonl`, mapping video **title → user turn**, video **transcript → assistant turn**, mirroring the app's real system prompt) → upload to Google Drive (rclone or web UI) → `train_unsloth_sft.py` **on Google Colab (T4 GPU) or Linux+CUDA only — Unsloth does not run on macOS** → exports LoRA adapter + `q4_k_m` GGUF + `Modelfile` → download and `ollama create arabic-asmr-<dialect> -f Modelfile` → point the app at it via `.env` (`LLM_PROVIDER=ollama`, `OLLAMA_MODEL=arabic-asmr-<dialect>:latest`) → evaluate with `eval_ab.py` (A/B vs base model, writes `eval_report.md`).

Full step-by-step (including exact Colab commands and `--no-resume` semantics for checkpoint handling) lives in `finetuning/README.md` — read it before touching this sub-pipeline. Has its own `finetuning/tests/`.

## 7. Testing

Run all tests with:
```bash
uv run pytest
```
- `tests/` — top-level app smoke/regression tests: `test_app_smoke.py` (sanitize_text, prompt resolution, voice filtering), `test_content_fetcher_smoke.py`, `test_llm_smoke.py`, `test_stt_smoke.py`, `test_tts_smoke.py`, `test_tts_regressions.py`, `test_fish_tts_smoke.py`, `test_preprocess_arabic_smoke.py`, `test_arabic_features_smoke.py`.
- `finetuning/tests/` — `test_build_dataset.py`, `test_ingest.py`, `test_synthesize.py`, `test_train_args.py`.
- Tests rely heavily on `unittest.mock.patch` for network/model calls — no live API calls or GPU/model downloads should be required to run the suite.
- The README also documents **manual, end-to-end UI test cases** (English rain-whisper session, Arabic storytelling session) with exact UI steps and expected behaviors — these are not automated and should be run manually when validating pipeline changes.

## 8. Current Known Issues / Active Work (from `qa_test_issues.md`)

This file is the closest thing to a live issue tracker; treat it as authoritative context for "what's broken / in progress" (note: gitignored, so check it directly, not git history):

1. **SILMA/F5-TTS removal**: the SILMA/F5-TTS Arabic approach was deemed unsatisfactory and should be (or has been, verify current state) fully removed from app logic and README, keeping **only** Fish Audio `s2.1-pro-free` for Arabic.
2. **Arabic transcript quality regression**: a quality drop was observed between two logged sessions (`logs/asmr_session_20260728_105819.log` good vs `logs/asmr_session_20260729_155408.log` regressed — sentences became too short/scattered). Root-cause and fix pending; compare against the good-quality log as ground truth.
3. **LLM provider selection nuance requested**: Arabic transcript generation should always default to `GEMINI_MODEL` regardless of `LLM_PROVIDER`; English transcript generation should follow `LLM_PROVIDER` as selected by the user. (Current `llm.py` forces Gemini for Arabic — verify this matches the intended semantics precisely, i.e. specifically the *model* configured in `GEMINI_MODEL`.)
4. **Arabic audio hallucination**: Fish Audio output has good whisper quality but audible hallucinated words (out-of-scope of the transcript), especially in the first half of sessions — needs isolation of whether the cause is transcript-side or audio-synthesis-side, with a green test once fixed.
5. **Voice naming/catalog change**: the app now uses Fish Audio for **both** English and Arabic, with display names sourced from `fish-audio-models.txt`. The default voice per language is the first Whispering voice (Arabic = **"ASMR 1 (Arabic Female)"**, `0de68eaa0cc5438389b82bba728c8e39`; English = **"ASMR English Female 5"**, `efee6c804185420fb89955186451df85`). The real Fish Audio profile names must **not** be surfaced in the app UI — only the app display names.

When asked to work on Arabic TTS/LLM quality, **always check `qa_test_issues.md` first** for the latest state of these items before assuming they're resolved or unresolved.

## 9. Conventions & Gotchas

- **Package manager is `uv`**, not pip/poetry/conda. Use `uv add`, `uv sync`, `uv run`.
- **Unsloth/QLoRA training code (`finetuning/train_unsloth_sft.py`) only runs on Linux+CUDA (or Colab)** — never attempt to run or debug it as if it works on macOS; it will fail at import. The Mac side only builds datasets and later consumes the exported GGUF via Ollama.
- **`torchcodec` is intentionally disabled** in `stt.py` (`_disable_torchcodec_for_asr`) to avoid a known crash (`Could not load libtorchcodec`) — don't remove this without addressing the underlying crash.
- **Silence markers (`<<SILENCE<n>MS>>`)** are an internal contract between `app.sanitize_text()`, `preprocess_arabic.normalize_arabic()`/`diacritize()` (which must pass them through untouched), and `tts.synthesize()`/`tts.generate_fish_audio()` (which convert them to actual silence or Fish Audio pause tags). If you add new text-processing steps in the pipeline, they must preserve these markers verbatim.
- **Both English and Arabic are routed through Fish Audio only** (all voice IDs are prefixed `fish_`). Do not reintroduce Kokoro, Edge-TTS, or F5-TTS/SILMA-based synthesis (explicitly removed per QA).
- **Never commit secrets**: `.env` is gitignored; only `.env.example` (with placeholder values) should be edited/tracked.
- **Almost all Markdown is gitignored** except `README.md` and `finetuning/README.md` — if you create new documentation intended to be tracked, either name it accordingly or update `.gitignore`.
- Logs (`logs/asmr_session_<session_id>.log`) capture the transcribed/typed prompt, raw LLM script, and sanitized TTS script per session — these are the primary debugging artifact for script-quality issues (see §8 item 2).
- The app is a single-file Streamlit orchestrator (`app.py`); it self-bootstraps via `os.system("python -m streamlit run ...")` when executed directly with plain `python`, so `uv run python app.py` is the correct invocation, not `streamlit run app.py` directly (though that also works once dependencies are installed).
