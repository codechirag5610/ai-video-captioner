"""Local LLM-judge — a stand-in for the hackathon's judge.

Scores each caption on ACCURACY (does it faithfully reflect the clip?) and TONE
(does it truly read as its style?), and runs a blind tech-vs-non-tech
distinguishability check. Drives the self-critique loop and the offline eval.

Use a DIFFERENT model here than the generator to avoid self-preference bias
(set judge.model != style.model in config/models.yaml).
"""
from __future__ import annotations

import logging
from typing import Any

from .client import FireworksClient
from .config import ModelSpec
from .styles import STYLE_LABELS, STYLES

log = logging.getLogger("captioner.judge")

SYSTEM = (
    "You are a strict, fair judge of video captions. You are given a factual "
    "description of a clip and four captions written in four styles. Score each "
    "on two axes from 1 to 10 and be discriminating -- reserve 9-10 for captions "
    "that are both accurate and nail the style. Return strict JSON."
)

_STYLE_TONE_CRITERIA = {
    "formal": "objective, complete sentences, no jokes, no contractions, news-summary register",
    "sarcastic": "dry deadpan irony aimed at real clip events, not mean, not random",
    "humorous-tech": "genuinely funny AND identifiably about programming/IT/engineering",
    "humorous-non-tech": "genuinely funny for a general audience with ZERO tech references",
}


def _build_prompt(gt_text: str, captions: dict[str, str]) -> str:
    crit = "\n".join(f'- "{k}": {v}' for k, v in _STYLE_TONE_CRITERIA.items())
    caps = "\n".join(f'- "{STYLE_LABELS[s]}": {captions.get(STYLE_LABELS[s], "")!r}' for s in STYLES)
    return (
        f"CLIP DESCRIPTION (ground truth):\n{gt_text}\n\n"
        f"CAPTIONS TO JUDGE:\n{caps}\n\n"
        "For EACH style score:\n"
        "  accuracy (1-10): does the caption faithfully reflect the clip and avoid "
        "invented facts?\n"
        "  tone (1-10): does it match its required style below?\n"
        f"Style tone requirements:\n{crit}\n\n"
        "Also judge: reading ONLY the two humorous captions WITHOUT their labels, "
        "could you reliably tell which is the tech one? Report 'distinguishable': "
        "true/false and 'nontech_has_tech_words': true/false (does humorous-non-tech "
        "contain any technology vocabulary?).\n\n"
        "Return ONLY JSON of the form:\n"
        "{\n"
        '  "formal": {"accuracy": n, "tone": n, "critique": "short reason"},\n'
        '  "sarcastic": {"accuracy": n, "tone": n, "critique": "..."},\n'
        '  "humorous-tech": {"accuracy": n, "tone": n, "critique": "..."},\n'
        '  "humorous-non-tech": {"accuracy": n, "tone": n, "critique": "..."},\n'
        '  "distinguishable": true,\n'
        '  "nontech_has_tech_words": false\n'
        "}"
    )


def _num(v: Any, default: float = 5.0) -> float:
    try:
        return max(1.0, min(10.0, float(v)))
    except (TypeError, ValueError):
        return default


def judge(
    gt_text: str,
    captions: dict[str, str],
    spec: ModelSpec,
    client: FireworksClient,
) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": _build_prompt(gt_text, captions)},
    ]
    raw = client.chat_json(spec, messages, temperature=0.0)

    scores: dict[str, dict[str, Any]] = {}
    totals = []
    for s in STYLES:
        label = STYLE_LABELS[s]
        entry = raw.get(label, {}) if isinstance(raw.get(label), dict) else {}
        acc = _num(entry.get("accuracy"))
        tone = _num(entry.get("tone"))
        scores[label] = {
            "accuracy": acc,
            "tone": tone,
            "critique": str(entry.get("critique", "")).strip(),
        }
        totals.append((acc + tone) / 2.0)

    distinguishable = bool(raw.get("distinguishable", True))
    nontech_leak = bool(raw.get("nontech_has_tech_words", False))

    # Penalize the overall a touch when the two humor styles blur or non-tech leaks.
    overall = sum(totals) / len(totals) if totals else 0.0
    if not distinguishable:
        overall -= 0.5
    if nontech_leak:
        overall -= 0.5

    return {
        "scores": scores,
        "distinguishable": distinguishable,
        "nontech_has_tech_words": nontech_leak,
        "overall": round(overall, 2),
    }
