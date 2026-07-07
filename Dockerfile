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

# App code.
COPY pyproject.toml ./
COPY src/ ./src/
COPY data/ ./data/
COPY config/ ./config/

ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1

# Default: read clips from /app/clips, write /app/output/captions.json.
# Mount your data:  docker run --rm --env-file .env \
#   -v $PWD/clips:/app/clips -v $PWD/output:/app/output -v $PWD/cache:/app/cache captioner
ENTRYPOINT ["python", "-m", "captioner.cli"]
CMD ["--input", "/app/clips", "--output", "/app/output/captions.json"]
