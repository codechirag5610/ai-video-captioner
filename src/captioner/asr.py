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
from pathlib import Path
from typing import Any

from .client import FireworksClient
from .config import AsrConfig

log = logging.getLogger("captioner.asr")

# Cache loaded models by size so a batch loads the (heavy) model once.
_local_models: dict[str, Any] = {}


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


def _load_model(size: str):
    """Load (and cache) a faster-whisper model, preferring GPU/float16 and
    falling back to CPU/int8 so it runs anywhere."""
    if size in _local_models:
        return _local_models[size]
    from faster_whisper import WhisperModel  # heavy dep; imported lazily

    last_err: Exception | None = None
    for device, compute in (("cuda", "float16"), ("cpu", "int8")):
        try:
            log.info("loading faster-whisper '%s' (device=%s, compute=%s)...", size, device, compute)
            model = WhisperModel(size, device=device, compute_type=compute)
            _local_models[size] = model
            return model
        except Exception as e:  # no CUDA / unsupported compute -> try next
            last_err = e
            log.debug("faster-whisper load failed on %s/%s: %s", device, compute, e)
    raise RuntimeError(f"could not load faster-whisper model '{size}': {last_err}")


def _transcribe_local(audio_path: Path, size: str) -> dict[str, Any]:
    model = _load_model(size)
    # vad_filter drops long silences -> less hallucination on quiet clips.
    segments, info = model.transcribe(
        str(audio_path),
        vad_filter=True,
        beam_size=5,
        condition_on_previous_text=False,  # avoids runaway repetition on short clips
    )
    text = " ".join(seg.text.strip() for seg in segments).strip()
    lang = getattr(info, "language", None)
    prob = getattr(info, "language_probability", None)
    log.info(
        "ASR: %d chars, language=%s%s",
        len(text), lang, f" (p={prob:.2f})" if isinstance(prob, float) else "",
    )
    return {"text": text, "language": lang}
