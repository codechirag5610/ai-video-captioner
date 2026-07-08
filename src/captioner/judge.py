"""Stage 5: Judge & select (the best-of-N engine).

For one style, score every candidate against the fact sheet on the same axes the
real LLM-judge cares about -- accuracy and tone -- plus distinctness from the
already-chosen styles and fit. Return the winning candidate and, when even the
best one is weak, a critique to drive one bounded regeneration.

Use a judge model from a DIFFERENT family than the generator (set judge.model !=
style.model in config) to avoid tuning to one model's taste.
"""
from __future__ import annotations

import logging
from typing import Any

from .client import FireworksClient
from .config import ModelSpec
from .styles import STYLE_CARDS, STYLE_LABELS

log = logging.getLogger("captioner.judge")

SYSTEM = (
    "You are a strict, fair judge of short-video captions. You compare candidates "
    "against a ground-truth account and a style definition. Be discriminating: "
    "reserve 9-10 for captions that are both accurate and truly nail the style. "
    "Any invented detail (something not in the account) caps that candidate's "
    "accuracy at 3. Return strict JSON."
)


def _build_prompt(style_key: str, gt_text: str, candidates: list[str], others: dict[str, str]) -> str:
    numbered = "\n".join(f"{i+1}. {c}" for i, c in enumerate(candidates))
    others_block = (
        "\n".join(f'- {lbl}: "{cap}"' for lbl, cap in others.items() if cap)
        or "(none selected yet)"
    )
    fit_line = (
        "for the humor styles, does the joke land? for formal, is it clean, complete, and free of humor?"
    )
    return (
        f"GROUND-TRUTH ACCOUNT OF THE VIDEO:\n{gt_text}\n\n"
        f"STYLE DEFINITION:\n{STYLE_CARDS[style_key]}\n\n"
        f"ALREADY-CHOSEN CAPTIONS IN OTHER STYLES (for the distinctness check):\n{others_block}\n\n"
        f"CANDIDATES:\n{numbered}\n\n"
        "Score EACH candidate 1-10 on:\n"
        "  accuracy: consistent with the account; asserts nothing not in it (any invented detail caps this at 3).\n"
        "  tone: matches the style definition's voice; would a reader label it as this style?\n"
        "  distinct: clearly different in voice from the already-chosen captions above.\n"
        f"  fit: {fit_line}\n\n"
        "Then pick the best candidate overall (prioritize accuracy, then tone).\n"
        "Return ONLY JSON:\n"
        "{\n"
        '  "candidates": [{"accuracy": n, "tone": n, "distinct": n, "fit": n, "justification": "one line"}],\n'
        '  "winner": <1-based index of the best candidate>,\n'
        '  "regenerate": "<what to fix, or empty string if the winner is strong>"\n'
        "}"
    )


def _num(v: Any, default: float = 5.0) -> float:
    try:
        return max(1.0, min(10.0, float(v)))
    except (TypeError, ValueError):
        return default


def _composite(s: dict[str, float]) -> float:
    # Accuracy and tone are what the real judge scores; weight them; distinct/fit break ties.
    return s["accuracy"] * 1.0 + s["tone"] * 1.0 + s["distinct"] * 0.5 + s["fit"] * 0.5


def select_best(
    style_key: str,
    gt_text: str,
    candidates: list[str],
    others: dict[str, str],
    spec: ModelSpec,
    client: FireworksClient,
    min_score: float = 7.0,
) -> dict[str, Any]:
    """Score candidates and return the winner + whether regeneration is warranted."""
    candidates = [c for c in candidates if c.strip()]
    if not candidates:
        return {
            "winner": "", "winner_index": -1,
            "accuracy": 0.0, "tone": 0.0, "distinct": 0.0, "fit": 0.0,
            "critique": "No candidates were generated.", "needs_regen": True, "all_scores": [],
        }

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": _build_prompt(style_key, gt_text, candidates, others)},
    ]
    try:
        raw = client.chat_json(spec, messages, temperature=0.0)
    except Exception as e:
        log.warning("judge call failed for %s; falling back to first candidate: %s", style_key, e)
        return {
            "winner": candidates[0], "winner_index": 0,
            "accuracy": 5.0, "tone": 5.0, "distinct": 5.0, "fit": 5.0,
            "critique": "", "needs_regen": False, "all_scores": [],
        }

    # Defensive: the judge may return a top-level array, a scalar, or index-keyed
    # candidates ({"1": {...}}). Coerce to the expected shape rather than crash.
    if not isinstance(raw, dict):
        raw = {}
    raw_scores = raw.get("candidates")
    if isinstance(raw_scores, dict):
        # index-keyed object -> ordered list
        raw_scores = [raw_scores[k] for k in sorted(raw_scores, key=lambda x: str(x))]
    if not isinstance(raw_scores, list):
        raw_scores = []
    scores: list[dict[str, float]] = []
    for i in range(len(candidates)):
        entry = raw_scores[i] if i < len(raw_scores) and isinstance(raw_scores[i], dict) else {}
        scores.append({
            "accuracy": _num(entry.get("accuracy")),
            "tone": _num(entry.get("tone")),
            "distinct": _num(entry.get("distinct")),
            "fit": _num(entry.get("fit")),
            "justification": str(entry.get("justification", "")).strip(),
        })

    # Trust the judge's winner if the index is valid; else argmax composite.
    idx = raw.get("winner")
    try:
        idx = int(idx) - 1
    except (TypeError, ValueError):
        idx = -1
    if not (0 <= idx < len(candidates)):
        idx = max(range(len(candidates)), key=lambda i: _composite(scores[i]))

    win = scores[idx]
    judge_regen = str(raw.get("regenerate", "")).strip()
    weak = min(win["accuracy"], win["tone"]) < min_score
    critique = judge_regen or (win.get("justification", "") if weak else "")

    return {
        "winner": candidates[idx],
        "winner_index": idx,
        "accuracy": win["accuracy"],
        "tone": win["tone"],
        "distinct": win["distinct"],
        "fit": win["fit"],
        "critique": critique,
        "needs_regen": bool(weak or judge_regen),
        "all_scores": scores,
    }
