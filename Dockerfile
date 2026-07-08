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
ARG WHISPER_MODEL=small
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('${WHISPER_MODEL}', device='cpu', compute_type='int8')"

# App code.
COPY pyproject.toml ./
COPY src/ ./src/
COPY data/ ./data/
COPY config/ ./config/

ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1

# Track 2 injects NO credentials, so the Fireworks key must be baked at build time:
#   docker build --build-arg FIREWORKS_API_KEY=fw_... --platform linux/amd64 -t <img> .
# WARNING: this key becomes readable in the public image (docker inspect / history).
# Use a spend-capped, rotatable key and rotate it after the event.
ARG FIREWORKS_API_KEY=""
ENV FIREWORKS_API_KEY=$FIREWORKS_API_KEY

# Judging-harness entrypoint: reads /input/tasks.json, downloads each video_url,
# captions it, writes /output/results.json. Override paths with INPUT_PATH/OUTPUT_PATH.
ENTRYPOINT ["python", "-m", "captioner.harness"]
