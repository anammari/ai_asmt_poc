# AI ASMR Proof of Concept

An automated AI-driven pipeline designed to fetch web content or direct topic prompts, transform them into soothing, whisper-style scripts using LLMs, and synthesize them into relaxing ASMR audio using neural Text-to-Speech (TTS) engines.

## Features

- **Content Ingestion** (`content_fetcher.py`): Fetches raw text or web articles directly from URLs.
- **LLM Script Rewriting** (`llm.py`): Rewrites source material into calm, rhythmic, whispered scripts with auditory markers (`[soft breath]`, `[whisper]`). Supports Google Gemini or local Ollama models.
- **Whisper Audio Synthesis** (`tts.py`): Synthesizes whisper-toned `.wav`/`.mp3` audio files using Kokoro-82M or Edge-TTS.
- **Audio Processing** (`stt.py`): Transcribes audio inputs using Whisper for voice-guided feeds.

## Environment Configuration

Before running locally or via Docker, copy `.env.example` to `.env` and set your credentials:

```bash
cp .env.example .env
```

Key `.env` settings:

```env
# LLM Provider Selection: "gemini", "openai", or "ollama"
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash-lite
GEMINI_FALLBACK_MODELS=gemini-2.5-flash,gemini-2.0-flash,gemini-flash-latest

# TTS Engine Selection: "kokoro" or "edge-tts"
TTS_ENGINE=kokoro
KOKORO_VOICE=af_bella

# Audio Output Path
OUTPUT_DIR=./output
```

Ollama inside Docker note:

If you are running the app inside Docker and Ollama natively on your Mac/Linux machine, set:

```env
OLLAMA_BASE_URL=http://host.docker.internal:11434
```

## Option 1: Running with Docker (Recommended)

Docker packages all system dependencies (`espeak-ng`, `ffmpeg`, Python 3.12).

### 1. Build the Docker image

```bash
docker build -t ai-asmr-poc .
```

### 2. Run the container

Run the container using your `.env` configuration and mount the `./output` directory so generated `.wav` audio files persist on your machine:

```bash
docker run --rm \
  --env-file .env \
  -v $(pwd)/output:/app/output \
  -p 8000:8000 \
  ai-asmr-poc
```

## Option 2: Running Natively on Local Host

### 1. Prerequisites and system dependencies

Ensure Python 3.12, `uv`, and `espeak-ng` (required phonemizer) are installed on your system:

macOS:

```bash
brew install espeak-ng
```

Ubuntu/Debian:

```bash
sudo apt-get update && sudo apt-get install -y espeak-ng ffmpeg
```

### 2. Install dependencies

```bash
uv sync
```

### 3. Optional warmup / pre-download TTS weights

You can optionally pre-download the Kokoro-82M model assets to your Hugging Face cache prior to running the app:

```bash
uv run python -c "from tts import create_tts_pipeline; create_tts_pipeline()"
```

### 4. Run application

```bash
uv run python app.py
```

## Test Case: Rain Whisper Session (English)

Use the following prompt to validate an end-to-end English run:

```text
Explain the soothing sound of gentle rain falling on a window and a show relaxing whisper.
```

### Manual test steps

1. Start the app:

```bash
uv run python app.py
```

2. In the UI:
  - (Optional) Add URL context in **Source URL Context (Optional)**.
  - Set **Script Language** to **English**.
  - Record instructions in **Voice Prompt (Instruct the ASMR topic & style)**.
  - (Optional) Fill **Optional: About You (Name, age, work, hobbies, etc.)**, for example:
    `My name is John. I have two children and I often read them bedtime stories.`
  - Select at least **2 voices** in **Select Voice(s) (Will alternate per paragraph)** to validate voice alternation.
  - Set **Vocal Tone Preference** to **Whispering**.
  - Set **Target Duration (Minutes)** to **3**.
  - Click **✨ Generate New ASMR**.

3. Verify expected behavior:
  - The app shows **Transcribed Instructions** based on your voice recording.
  - The generated script is sanitized before synthesis (formatting artifacts removed, pause tags converted to `<<SILENCE...MS>>` markers).
  - If Gemini returns transient 429/500/503 errors, generation uses exponential backoff retries, then tries fallback Gemini models, then falls back to Ollama (or local minimal fallback if Ollama is unavailable).
  - **View TTS-Ready Script** displays human-readable silence hints like *(Silence: 3000 ms)*.
  - Output audio plays in-app and includes real silent gaps where pause markers exist.
  - Multi-voice selection alternates voices across script paragraphs.
  - **🔄 Re-Synthesize (Use Existing Script)** regenerates audio from the cached sanitized script without re-calling the LLM.
  - A session log is written to `logs/asmr_session_<session_id>.log` including transcribed prompt, raw script, and sanitized script.

## Test Case: جلسة المطر الهادئ (Arabic Rain Session)

Use the following prompt to validate an end-to-end Arabic run:

```text
صِف لي صوتاً هادئاً للمطر يلمس النافذة ببطء، وارسم لي مشهداً من الهدوء والسكينة لكي أنام
```

### Manual test steps

1. Start the app:

```bash
uv run python app.py
```

2. In the UI:
  - (Optional) Add URL context in **Source URL Context (Optional)**.
  - Set **Script Language** to **Arabic (العربية)**.
  - Enter the Arabic prompt above in **Arabic Written Prompt (النص العربي)** (voice input is not used).
  - (Optional) Fill **Optional: About You (Name, age, work, hobbies, etc.)** for personalized delivery.
  - For **Vocal Tone Preference**, select **Whispering** (shows ASMR 1 & ASMR 2 voices) or **Soft Spoken** (shows ASMR 3 & ASMR 4 voices).
  - Select one or more voices from the available Arabic voice list; voices alternate per paragraph.
  - Set **Target Duration (Minutes)** to **3**.
  - Click **✨ Generate New ASMR**.

3. Verify expected behavior:
  - The app shows **Arabic Written Instructions** with your input text.
  - The generated script is in fluent Arabic with 4-6 paragraphs separated by blank lines and varied pause durations (`[pause]`, `[pause:2s]`, `[pause:3s]`, `[pause:4s]`).
  - **View TTS-Ready Script** displays human-readable silence hints.
  - Output audio is a clear Arabic whispered or soft-spoken ASMR track with no hallucinated words.
  - Multi-voice selection alternates voices across paragraphs.
  - **🔄 Re-Synthesize (Use Existing Script)** regenerates audio without re-calling the LLM.
  - A session log is written to `logs/asmr_session_<session_id>.log`.

## Arabic ASMR Toolkit

### 1. Fish Audio TTS backend (`fish_audio_tts.py`)

The sole Arabic TTS backend. Uses Fish Audio `s2.1-pro-free` with inline
S2 direction tags (`[whispering]`, `[soft]`, `[break]`). The app provides
4 community ASMR voices filtered by vocal tone. Evaluate it standalone:

```bash
# set FISH_AUDIO_API_KEY in .env first
python fish_audio_tts.py
# -> output/output_asmr_arabic.mp3 (+ per-case files), output/fish_audio_assessment.md
```

### 5. Arabic transcript fine-tuning (`finetuning/`)

Unsloth QLoRA fine-tune of an Arabic-capable instruct model on curated
Arabic ASMR scripts, served back to the app via Ollama. Full developer
action points: [`finetuning/README.md`](finetuning/README.md).

```bash
uv run python finetuning/build_dataset.py --count 60   # local, needs GEMINI_API_KEY
# then train on Colab (T4), export GGUF, `ollama create arabic-asmr -f Modelfile`
# and set LLM_PROVIDER=ollama + OLLAMA_MODEL=arabic-asmr:latest
```

## Project Structure

```text
ai_asmr_poc/
|- README.md                         # Project setup, run guide, and validation steps.
|- app.py                            # Streamlit UI entrypoint and orchestration flow.
|- content_fetcher.py                # Fetches source context and handles fallback behavior.
|- llm.py                            # Local Ollama rewrite client with endpoint fallback.
|- stt.py                            # Whisper-based speech-to-text preprocessing and inference.
|- tts.py                            # Kokoro + Fish Audio TTS synthesis with silence marker support.
|- preprocess_arabic.py              # Arabic normalization + diacritization (tashkeel).
|- fish_audio_tts.py                 # Fish Audio s2.1-pro-free Arabic ASMR TTS client.
|- finetuning/                       # Unsloth Arabic ASMR SFT pipeline (see finetuning/README.md).
|- Dockerfile                        # Container build with runtime dependencies.
|- pyproject.toml                    # Python project metadata and dependencies.
|- uv.lock                           # Locked dependency resolution for reproducible installs.
|- .env.example                      # Template environment variables for local setup.
|- .gitignore                        # Git ignore patterns for generated/local files.
|- .python-version                   # Preferred Python interpreter version.
|- logs/                             # Per-session generation and synthesis logs.
|- tests/                            # Smoke tests for app, LLM, fetcher, STT, and TTS flows.
   |- test_app_smoke.py              # Validates app-side helpers and text preparation.
   |- test_content_fetcher_smoke.py  # Validates context fetch fallback behavior.
   |- test_llm_smoke.py              # Validates local LLM endpoint fallback behavior.
   |- test_stt_smoke.py              # Validates audio transcription input handling.
   |- test_tts_smoke.py              # Validates TTS API compatibility and synthesis path.
   |- test_fish_tts_smoke.py         # Validates Fish Audio client, fallback, app routing.
   `- test_preprocess_arabic_smoke.py # Validates Arabic normalization/diacritization behavior.
```
