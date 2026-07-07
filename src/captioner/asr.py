"""Audio transcription. Pluggable backend: fireworks | local | none."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .client import FireworksClient
from .config import AsrConfig

log = logging.getLogger("captioner.asr")

_local_model = None  # lazily loaded faster-whisper model


def transcribe(audio_path: Path | None, cfg: AsrConfig, client: FireworksClient) -> dict[str, Any]:
    """Return {text, language}. Empty text means silent / no speech."""
    empty = {"text": "", "language": None}
    if audio_path is None or cfg.backend == "none":
        return empty

    try:
        if cfg.backend == "fireworks":
            return client.transcribe(str(audio_path), cfg.model)
        if cfg.backend == "local":
            return _transcribe_local(audio_path, cfg.local_model_size)
    except Exception as e:
        log.warning("ASR failed (continuing without transcript): %s", e)
        return empty

    log.warning("unknown ASR backend %r; skipping", cfg.backend)
    return empty


def _transcribe_local(audio_path: Path, size: str) -> dict[str, Any]:
    global _local_model
    if _local_model is None:
        from faster_whisper import WhisperModel  # optional dep
        _local_model = WhisperModel(size, device="auto", compute_type="int8")
    segments, info = _local_model.transcribe(str(audio_path), vad_filter=True)
    text = " ".join(seg.text.strip() for seg in segments).strip()
    return {"text": text, "language": getattr(info, "language", None)}
