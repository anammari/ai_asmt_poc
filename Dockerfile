# Use Python 3.12 slim image matching .python-version
FROM python:3.12-slim

# Prevent Python from writing bytecode files and enable unbuffered logging
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install required system dependencies (espeak-ng for phonemization, ffmpeg for audio)
RUN apt-get update && apt-get install -y --no-install-recommends \
    espeak-ng \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set container working directory
WORKDIR /app

# Copy official `uv` binary from Astral's image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy dependency configuration files first for Docker layer caching
COPY pyproject.toml uv.lock ./

# Install project dependencies
RUN uv sync --frozen

# Explicitly set Hugging Face cache path inside container
ENV HF_HOME=/app/.cache/huggingface

# Copy the rest of the application codebase
COPY . .

# =========================================================
# BUILD-TIME MODEL PRE-LOADING
# Bakes ~327MB Kokoro-82M model weights directly into Docker image
# =========================================================
RUN uv run python tts.py --preload

# Ensure output directory exists for exported audio files
RUN mkdir -p /app/output

# Expose default application port
EXPOSE 8000

# Execute main application orchestrator
CMD ["uv", "run", "python", "app.py"]