# Slim, reproducible image. Python 3.11 has the best wheel coverage for our deps.
FROM python:3.11-slim

# ffmpeg + ffprobe are the only system deps we need.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first for better layer caching.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the faster-whisper model into the image so runs never stall on a
# first-use download (and work even if HF Hub is slow/unreachable at runtime).
# Keep WHISPER_MODEL in sync with asr.local_model_size in config/models.yaml.
# HF_HOME pins the cache inside the image so it survives any runtime user/home.
ENV HF_HOME=/app/.hf
ARG WHISPER_MODEL=base
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('${WHISPER_MODEL}', device='cpu', compute_type='int8')"

# App code.
COPY pyproject.toml ./
COPY src/ ./src/
COPY data/ ./data/
COPY config/ ./config/

ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1

# Track 2 injects NO credentials, so keys must be baked at build time:
#   docker build --build-arg FIREWORKS_API_KEY=fw_... \
#                --build-arg GEMINI_API_KEY=... [--build-arg OPENROUTER_API_KEY=...] \
#                --platform linux/amd64 -t <img> .
# WARNING: keys become readable in the public image (docker inspect / history).
# Use spend-capped, rotatable keys and rotate them after the event.
# GEMINI_API_KEY powers the Gemma 4 stages (vision fact-sheet + all caption
# writing); when absent those stages fall back to Fireworks automatically.
ARG FIREWORKS_API_KEY=""
RUN test -n "$FIREWORKS_API_KEY" || (echo "FIREWORKS_API_KEY build-arg is required" && exit 1)
ENV FIREWORKS_API_KEY=$FIREWORKS_API_KEY
ARG GEMINI_API_KEY=""
ENV GEMINI_API_KEY=$GEMINI_API_KEY
ARG OPENROUTER_API_KEY=""
ENV OPENROUTER_API_KEY=$OPENROUTER_API_KEY

# Judging-harness entrypoint: reads /input/tasks.json, downloads each video_url,
# captions it, writes /output/results.json. Override paths with INPUT_PATH/OUTPUT_PATH.
ENTRYPOINT ["python", "-m", "captioner.harness"]
