"""Stage A: Understanding.

Extract a dense, NEUTRAL, factual ground-truth description from the sampled
frames + transcript. This stage owns the pipeline's ACCURACY score. It makes no
jokes and takes no stylistic stance -- it just reports what is in the clip.

The `notable` field ("the single most surprising/funny/ironic thing") is the raw
material Stage B turns into humor and sarcasm, so tone stays anchored to facts.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .cache import Cache
from .client import FireworksClient
from .config import ModelSpec
from .preprocess import Preprocessed, encode_frame

log = logging.getLogger("captioner.understand")

# Ground-truth JSON contract. Every field is required (may be empty string/list).
GROUND_TRUTH_KEYS = [
    "setting", "subjects", "events", "dialogue_summary", "visible_text",
    "audio_description", "mood", "notable", "confidence",
]

SYSTEM = (
    "You are a meticulous video analyst. You are shown frames sampled from a "
    "single short video clip (with timestamps), plus a transcript of its audio. "
    "Report ONLY what is actually observable. Do not invent details, names, or "
    "dialogue. If something is unclear, say so rather than guessing. Your output "
    "is the factual ground truth other systems depend on, so accuracy is "
    "everything."
)

INSTRUCTIONS = """From the frames and transcript, produce a JSON object with EXACTLY these keys:

- "setting": (string) where/when this takes place; environment, location, era if evident.
- "subjects": (array of strings) the people, animals, or main objects, with brief descriptors.
- "events": (array of strings) the ordered sequence of what happens, each item ideally prefixed with an approximate timestamp like "0:04 - ...". Base ordering on frame timestamps.
- "dialogue_summary": (string) what is said, summarized. Empty string if there is no speech.
- "visible_text": (string) transcribe any on-screen text, captions, signs, UI text, or memes exactly. Empty string if none.
- "audio_description": (string) non-speech audio: music mood, sound effects, silence. Do NOT transcribe song lyrics; describe the music instead.
- "mood": (string) the overall tone/emotion of the clip (e.g. tense, playful, chaotic, wholesome).
- "notable": (string) the single most surprising, funny, ironic, or attention-grabbing thing about the clip -- the "point" of it.
- "confidence": (number 0-1) how confident you are that the above is accurate given the available frames/audio. Lower it when frames are dark, blurry, few, or ambiguous.

Return ONLY the JSON object, no prose."""


def _sampling_signature(spec: ModelSpec, n_frames: int) -> str:
    return f"{spec.model}|imgs={n_frames}|edge={spec.image_max_edge}|q={spec.image_quality}"


def build_messages(pre: Preprocessed, transcript: dict[str, Any], spec: ModelSpec) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []

    # Context header: duration, audio state, transcript.
    dur = pre.probe.duration
    lang = transcript.get("language")
    text = (transcript.get("text") or "").strip()
    header = [f"Clip duration: {dur:.0f} seconds.", f"Frames provided: {len(pre.frames)} (timestamps shown per image)."]
    if not pre.probe.has_audio:
        header.append("This clip has NO audio track (silent). Do not describe speech or music.")
    elif not text:
        header.append("Audio is present but no speech was transcribed (likely music/ambient/no dialogue).")
    else:
        header.append(f"Audio transcript{f' (language: {lang})' if lang else ''}:\n\"\"\"\n{text}\n\"\"\"")
    content.append({"type": "text", "text": "\n".join(header)})

    # Frames, each preceded by its timestamp label.
    for fr in pre.frames:
        tag = f"Frame at {int(fr.timestamp // 60)}:{int(fr.timestamp % 60):02d}"
        if fr.is_scene_change:
            tag += " (scene change)"
        content.append({"type": "text", "text": tag})
        data_uri = encode_frame(
            fr.path,
            max_edge=spec.image_max_edge,
            fmt=spec.image_format,
            quality=spec.image_quality,
        )
        content.append({"type": "image_url", "image_url": {"url": data_uri}})

    content.append({"type": "text", "text": INSTRUCTIONS})
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": content},
    ]


def _normalize(gt: dict[str, Any]) -> dict[str, Any]:
    """Coerce model output into the fixed contract."""
    out: dict[str, Any] = {}
    for k in GROUND_TRUTH_KEYS:
        v = gt.get(k)
        if k == "subjects" or k == "events":
            if isinstance(v, str):
                v = [v] if v.strip() else []
            out[k] = [str(x) for x in (v or [])]
        elif k == "confidence":
            try:
                out[k] = max(0.0, min(1.0, float(v)))
            except (TypeError, ValueError):
                out[k] = 0.5
        else:
            out[k] = str(v).strip() if v is not None else ""
    return out


def understand(
    pre: Preprocessed,
    transcript: dict[str, Any],
    spec: ModelSpec,
    client: FireworksClient,
    cache: Cache | None = None,
    vhash: str | None = None,
) -> dict[str, Any]:
    sig = _sampling_signature(spec, len(pre.frames))
    if cache and vhash:
        hit = cache.get(vhash, "understand", sig)
        if hit:
            log.info("Stage A cache hit for %s", pre.video_path.name)
            return hit

    if not pre.frames:
        log.warning("no frames for %s; producing degraded ground truth", pre.video_path.name)
        gt = _normalize({
            "setting": "unknown", "mood": "unknown", "confidence": 0.1,
            "audio_description": transcript.get("text", ""),
            "dialogue_summary": transcript.get("text", ""),
            "notable": "Could not extract frames from this clip.",
        })
    else:
        messages = build_messages(pre, transcript, spec)
        raw = client.chat_json(spec, messages)
        gt = _normalize(raw)

    if cache and vhash:
        cache.put(vhash, "understand", sig, gt)
    return gt
