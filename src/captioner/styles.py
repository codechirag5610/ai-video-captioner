"""Stage B: Style rendering.

Turns the neutral ground truth into four captions in four distinct registers.
This stage owns the pipeline's TONE score.

Key design choices that win tone points:
  - All four are generated in ONE structured call so the model can actively
    DIFFERENTIATE them (esp. tech-humor vs non-tech-humor, which judges check).
  - Every style is required to reference >=2 concrete clip details, which keeps
    tone anchored to accuracy.
  - humorous_non_tech has an explicit BANNED tech-vocabulary list so it cannot
    blur into humorous_tech.
  - A single style can be regenerated with judge feedback (self-critique loop).
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

from .client import FireworksClient
from .config import ModelSpec

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data.style_examples import FEW_SHOT  # noqa: E402

log = logging.getLogger("captioner.styles")

STYLES = ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]

# Output labels the hackathon uses (hyphenated). Internal keys use underscores.
STYLE_LABELS = {
    "formal": "formal",
    "sarcastic": "sarcastic",
    "humorous_tech": "humorous-tech",
    "humorous_non_tech": "humorous-non-tech",
}

BANNED_TECH_WORDS = [
    "code", "coding", "deploy", "deployment", "server", "bug", "debug", "commit",
    "merge", "git", "CI", "CD", "pipeline", "build", "compile", "algorithm", "API",
    "database", "cloud", "software", "hardware", "app", "AI", "machine learning",
    "prod", "production", "backend", "frontend", "stack", "CPU", "GPU", "RAM",
    "null", "exception", "runtime", "refactor", "repo", "prompt", "token", "latency",
]

STYLE_SPECS = {
    "formal": (
        "FORMAL: Objective, precise, complete sentences in a neutral news-summary "
        "register. No contractions, no slang, no jokes, no opinion. This caption "
        "carries the accuracy score, so it must be factually exact. 1-2 sentences."
    ),
    "sarcastic": (
        "SARCASTIC: Dry, deadpan wit. Feign being unimpressed or praise the obvious "
        "as if it were genius; understate the chaos. Sarcastic, NOT mean or cruel, "
        "and NOT random -- the irony must target actual events in the clip. "
        "1-2 sentences."
    ),
    "humorous_tech": (
        "HUMOROUS-TECH: Genuinely funny using programming / IT / engineering culture "
        "-- map the clip onto things like deploys, merge conflicts, flaky tests, "
        "prod incidents, standups, CPU/RAM, null pointers, tech debt. The tech "
        "metaphor must fit what actually happens. Identifiably techie. 1-2 sentences."
    ),
    "humorous_non_tech": (
        "HUMOROUS-NON-TECH: Genuinely funny for a general audience with ZERO "
        "technology references. Everyday-life humor, relatable exaggeration, absurd "
        "comparisons. Must contain NO technology/computing vocabulary at all. "
        "1-2 sentences."
    ),
}

SYSTEM = (
    "You are an award-winning caption writer. Given a factual description of a "
    "short video clip, you write captions in several distinct styles. Rules that "
    "apply to EVERY style:\n"
    "  1. Each caption must clearly reference at least TWO specific things that "
    "actually happen in the clip (do not be generic).\n"
    "  2. Never invent facts that are not in the description.\n"
    "  3. Keep each caption to 1-2 sentences, punchy and self-contained.\n"
    "  4. The four styles must read as clearly different from one another; in "
    "particular, humorous-tech and humorous-non-tech must be unmistakably distinct."
)


def _gt_to_text(gt: dict[str, Any]) -> str:
    """Render ground truth as a compact briefing for Stage B."""
    lines = []
    if gt.get("setting"):
        lines.append(f"Setting: {gt['setting']}")
    if gt.get("subjects"):
        lines.append("Subjects: " + "; ".join(gt["subjects"]))
    if gt.get("events"):
        lines.append("Events:\n- " + "\n- ".join(gt["events"]))
    if gt.get("dialogue_summary"):
        lines.append(f"Dialogue: {gt['dialogue_summary']}")
    if gt.get("visible_text"):
        lines.append(f"On-screen text: {gt['visible_text']}")
    if gt.get("audio_description"):
        lines.append(f"Audio: {gt['audio_description']}")
    if gt.get("mood"):
        lines.append(f"Mood: {gt['mood']}")
    if gt.get("notable"):
        lines.append(f"Most notable thing: {gt['notable']}")
    conf = gt.get("confidence", 1.0)
    if conf < 0.5:
        lines.append(
            f"(NOTE: analysis confidence is low ({conf:.2f}). Only reference details "
            "you are given; do NOT invent specifics to be funny.)"
        )
    return "\n".join(lines)


def _few_shot_block() -> str:
    """Compact few-shot demonstrations covering all four styles."""
    blocks = []
    n_examples = len(next(iter(FEW_SHOT.values())))
    for i in range(n_examples):
        gt = FEW_SHOT["formal"][i][0]
        out = {STYLE_LABELS[s]: FEW_SHOT[s][i][1] for s in STYLES}
        blocks.append(
            f"EXAMPLE {i+1}\nCLIP DESCRIPTION: {gt}\nCAPTIONS:\n"
            + json.dumps(out, ensure_ascii=False, indent=2)
        )
    return "\n\n".join(blocks)


def _build_prompt(gt: dict[str, Any]) -> list[dict[str, Any]]:
    spec_lines = "\n".join(f"- {STYLE_SPECS[s]}" for s in STYLES)
    banned = ", ".join(BANNED_TECH_WORDS)
    label_keys = ", ".join(f'"{STYLE_LABELS[s]}"' for s in STYLES)
    user = (
        "Write one caption for the clip below in EACH of these four styles:\n"
        f"{spec_lines}\n\n"
        f"For humorous-non-tech, these words (and anything like them) are BANNED: {banned}.\n\n"
        "Here are worked examples of the four styles:\n\n"
        f"{_few_shot_block()}\n\n"
        "Now do the same for THIS clip.\n\n"
        f"CLIP DESCRIPTION:\n{_gt_to_text(gt)}\n\n"
        f"Return ONLY a JSON object with exactly these keys: {label_keys}. "
        "Each value is the caption string."
    )
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
    ]


def _sanitize_non_tech(caption: str) -> tuple[str, bool]:
    """Detect banned tech words in the non-tech caption. Returns (caption, clean)."""
    low = caption.lower()
    for w in BANNED_TECH_WORDS:
        # word-ish boundary check
        token = w.lower()
        if token in low:
            # avoid false positives like "cloud" in "clouds"? keep simple substring;
            # the regen loop will fix real leaks.
            import re
            if re.search(rf"\b{re.escape(token)}\b", low):
                return caption, False
    return caption, True


def generate(gt: dict[str, Any], spec: ModelSpec, client: FireworksClient) -> dict[str, str]:
    """Generate all four captions. Returns dict keyed by hyphenated style labels."""
    messages = _build_prompt(gt)
    raw = client.chat_json(spec, messages)

    out: dict[str, str] = {}
    for s in STYLES:
        label = STYLE_LABELS[s]
        val = raw.get(label) or raw.get(s) or ""
        out[label] = str(val).strip()

    # Flag non-tech leakage for the critique loop (don't hard-fail here).
    _, clean = _sanitize_non_tech(out[STYLE_LABELS["humorous_non_tech"]])
    if not clean:
        log.debug("non-tech caption contains banned tech vocabulary; critique loop should fix")
    return out


def regenerate_one(
    gt: dict[str, Any],
    style_key: str,
    critique: str,
    spec: ModelSpec,
    client: FireworksClient,
) -> str:
    """Regenerate a single style given judge feedback (self-critique loop)."""
    label = STYLE_LABELS[style_key]
    extra = ""
    if style_key == "humorous_non_tech":
        extra = f"\nBANNED words (do not use): {', '.join(BANNED_TECH_WORDS)}."
    user = (
        f"Rewrite ONLY the {label} caption for this clip. Style rules:\n"
        f"{STYLE_SPECS[style_key]}{extra}\n\n"
        f"Problem with the previous attempt: {critique}\n\n"
        f"CLIP DESCRIPTION:\n{_gt_to_text(gt)}\n\n"
        f'Return ONLY JSON: {{"{label}": "<the caption>"}}'
    )
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    raw = client.chat_json(spec, messages)
    return str(raw.get(label) or raw.get(style_key) or "").strip()
