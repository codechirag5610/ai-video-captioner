"""Stage 3: Comedy-material extraction.

A cheap LLM pass over the fact sheet that lists the inherently absurd / ironic /
notable elements and, for each, WHY it is funny and any tech-metaphor angle. This
separates *finding the funny* from *writing the joke*: the style generators then
aim at pre-validated, grounded material instead of free-associating -- which is
exactly where hallucinated humor creeps in.

Every element must cite the fact sheet, so nothing new is invented here.
Skipped for the formal style downstream.
"""
from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

from .cache import Cache
from .client import FireworksClient
from .config import ComedyConfig, ModelSpec

log = logging.getLogger("captioner.comedy")

SYSTEM = (
    "You are a comedy writer's researcher. Given a factual account of a short "
    "video, you identify what is inherently funny about it -- WITHOUT inventing "
    "anything. Every angle you list must be grounded in the account you are given."
)

INSTRUCTIONS = """List 3-6 elements of this video with comedic potential. Return a JSON object:
{
  "material": [
    {
      "element": "<a concrete thing from the account: an event, a detail, a contrast>",
      "why_funny": "<irony / absurdity / timing / relatability / scale-mismatch>",
      "tech_angle": "<a software/engineering concept it maps onto, or empty string if none fits>"
    }
  ]
}
Rules:
- Only use things present in the account. Do NOT invent details.
- Do NOT draw comedic angles from anything the account marks as uncertain.
- Prefer specific, grounded observations over generic ones.
Return ONLY the JSON object."""


def _fact_sheet_text(gt: dict[str, Any]) -> str:
    lines = []
    for label, key in [
        ("Setting", "setting"), ("Subjects", "subjects"), ("Events", "events"),
        ("Dialogue", "dialogue_summary"), ("On-screen text", "visible_text"),
        ("Audio", "audio_description"), ("Mood", "mood"), ("Most notable", "notable"),
    ]:
        v = gt.get(key)
        if isinstance(v, list):
            if v:
                lines.append(f"{label}: " + "; ".join(v))
        elif v:
            lines.append(f"{label}: {v}")
    if gt.get("uncertain"):
        lines.append("UNCERTAIN (do not build jokes on these): " + "; ".join(gt["uncertain"]))
    return "\n".join(lines)


def _render_material(material: list[dict[str, Any]]) -> str:
    """Flatten to a compact block injected into humor style prompts."""
    out = []
    for i, m in enumerate(material, 1):
        el = str(m.get("element", "")).strip()
        why = str(m.get("why_funny", "")).strip()
        tech = str(m.get("tech_angle", "")).strip()
        if not el:
            continue
        line = f"{i}. {el} — funny because {why}" if why else f"{i}. {el}"
        if tech:
            line += f" [tech angle: {tech}]"
        out.append(line)
    return "\n".join(out)


def extract_comedy(
    gt: dict[str, Any],
    cfg: ComedyConfig,
    style_spec: ModelSpec,
    client: FireworksClient,
    cache: Cache | None = None,
    vhash: str | None = None,
) -> dict[str, Any]:
    """Returns {"material": [...], "text": "<rendered block>"}.
    Empty material (disabled or failure) yields an empty text block."""
    if not cfg.enabled:
        return {"material": [], "text": ""}

    spec = style_spec if not cfg.model else replace(style_spec, model=cfg.model)
    sig = f"{spec.model}|comedy|t={cfg.temperature}"
    if cache and vhash:
        hit = cache.get(vhash, "comedy", sig)
        if hit is not None:
            return hit

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"FACTUAL ACCOUNT:\n{_fact_sheet_text(gt)}\n\n{INSTRUCTIONS}"},
    ]
    try:
        raw = client.chat_json(spec, messages, temperature=cfg.temperature)
        material = raw.get("material", [])
        if not isinstance(material, list):
            material = []
    except Exception as e:
        log.warning("comedy extraction failed (continuing without it): %s", e)
        material = []

    result = {"material": material, "text": _render_material(material)}
    if cache and vhash and material:
        cache.put(vhash, "comedy", sig, result)
    return result
