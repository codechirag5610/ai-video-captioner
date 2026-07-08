"""Audio transcription. Pluggable backend: local | fireworks | none.

Default is `local` (faster-whisper): a self-contained, offline, model-agnostic
Whisper that ships in the Docker image. Speech is the primary accuracy signal for
most clips, so this stage is first-class -- not an optional add-on.

`fireworks` remains available for accounts whose key is entitled to the Fireworks
audio/transcription product (this project's key is not -> 401), and `none`
disables ASR entirely (visual-only clips).
"""
from __future__ import annotations

import logging
import os
import re
import threading
from pathlib import Path
from typing import Any

from .client import FireworksClient
from .config import AsrConfig

log = logging.getLogger("captioner.asr")

# Cache loaded models by size so a batch loads the (heavy) model once. The lock
# stops parallel clip workers from racing 4 duplicate model loads in wave 1.
_local_models: dict[str, Any] = {}
_load_lock = threading.Lock()

# Whisper hallucinates YouTube-outro boilerplate on music/ambient clips; a fake
# "Thanks for watching!" asserted as dialogue poisons the fact sheet's accuracy.
_HALLUCINATION = re.compile(
    r"(?i)thanks?\s+for\s+watching|subscribe|like\s+and\s+|see you (in the )?next"
)


def transcribe(audio_path: Path | None, cfg: AsrConfig, client: FireworksClient) -> dict[str, Any]:
    """Return {text, language}. Empty text means silent / no speech / ASR off."""
    empty = {"text": "", "language": None}
    if audio_path is None or cfg.backend == "none":
        return empty

    try:
        if cfg.backend == "local":
            return _transcribe_local(audio_path, cfg.local_model_size)
        if cfg.backend == "fireworks":
            return client.transcribe(str(audio_path), cfg.model)
    except Exception as e:
        log.warning("ASR failed (continuing without transcript): %s", e)
        return empty

    log.warning("unknown ASR backend %r; skipping", cfg.backend)
    return empty


def preload(cfg: AsrConfig) -> None:
    """Load the whisper model once, before clip workers start, so wave 1 never
    races duplicate loads and the cost lands in the startup window."""
    if cfg.backend == "local":
        try:
            _load_model(cfg.local_model_size)
        except Exception as e:
            log.warning("ASR preload failed (will run without transcripts): %s", e)


def _load_model(size: str):
    """Load (and cache) a faster-whisper model. CPU/int8: the grading VM has no
    GPU, and probing CUDA just wastes startup seconds."""
    if size in _local_models:
        return _local_models[size]
    with _load_lock:
        if size in _local_models:  # double-checked: another worker won the race
            return _local_models[size]
        from faster_whisper import WhisperModel  # heavy dep; imported lazily

        log.info("loading faster-whisper '%s' (cpu/int8)...", size)
        model = WhisperModel(
            size, device="cpu", compute_type="int8",
            cpu_threads=max(1, (os.cpu_count() or 2) // 4),
        )
        _local_models[size] = model
        return model


def _transcribe_local(audio_path: Path, size: str) -> dict[str, Any]:
    model = _load_model(size)
    # vad_filter drops long silences -> less hallucination on quiet clips.
    # beam_size=1: greedy is ~2-3x faster and the transcript is a garnish on the
    # fact sheet, not the dish.
    segments, info = model.transcribe(
        str(audio_path),
        vad_filter=True,
        beam_size=1,
        condition_on_previous_text=False,  # avoids runaway repetition on short clips
    )
    kept = []
    for seg in segments:
        if getattr(seg, "no_speech_prob", 0.0) > 0.5:
            continue
        if getattr(seg, "avg_logprob", 0.0) < -1.0:
            continue
        kept.append(seg.text.strip())
    text = " ".join(kept).strip()
    # Boilerplate whole-transcript hallucinations: better no transcript than a lie.
    if text and (_HALLUCINATION.search(text) and len(text.split()) < 12):
        log.info("ASR: dropping hallucinated boilerplate transcript %r", text[:60])
        text = ""
    lang = getattr(info, "language", None)
    prob = getattr(info, "language_probability", None)
    log.info(
        "ASR: %d chars, language=%s%s",
        len(text), lang, f" (p={prob:.2f})" if isinstance(prob, float) else "",
    )
    return {"text": text, "language": lang}
