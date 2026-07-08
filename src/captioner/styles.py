"""Stage 4: Style generation (best-of-N).

For each of the four styles we generate N candidate captions from the SAME fact
sheet (never from raw frames) using a per-style "style card" with hand-written
exemplars and per-style temperature. The judge (Stage 5) then selects the best
candidate per style. This is the engine: variance across candidates is where the
good jokes come from; the judge keeps only the ones that land.

Guardrails baked into the prompts:
  - every caption must reference concrete FACTS (accuracy anchored to tone)
  - nothing from the fact sheet's `uncertain` list may be used
  - humorous_non_tech has an explicit banned tech-vocabulary list
  - formal skips comedic material entirely
"""
from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

from .client import FireworksClient
from .config import ModelSpec

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data.style_examples import FEW_SHOT  # noqa: E402

log = logging.getLogger("captioner.styles")

STYLES = ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]

# Output keys MUST match the Track 2 guide exactly (underscores). The guide's
# output example is {formal, sarcastic, humorous_tech, humorous_non_tech}; a
# hyphenated key reads as a MISSING style and scores zero for that clip.
STYLE_LABELS = {
    "formal": "formal",
    "sarcastic": "sarcastic",
    "humorous_tech": "humorous_tech",
    "humorous_non_tech": "humorous_non_tech",
}

# Styles that receive the Stage 3 comedic material (formal stays straight).
HUMOR_STYLES = {"sarcastic", "humorous_tech", "humorous_non_tech"}

BANNED_TECH_WORDS = [
    # Unambiguous tech terms. Homographs with common non-tech senses (bug=insect,
    # build, stack, commit, merge, cloud, token, prompt) are deliberately EXCLUDED
    # so the guard never rewrites a correct non-tech caption about an actual bug,
    # sandcastle build, stack of pancakes, or cloud in the sky.
    "code", "coding", "deploy", "deployment", "server", "debug",
    "git", "CI", "CD", "pipeline", "compile", "algorithm", "API",
    "database", "software", "hardware", "app", "AI", "machine learning",
    "prod", "backend", "frontend", "CPU", "GPU", "RAM",
    "null", "exception", "runtime", "refactor", "repo", "latency",
    "download", "upload", "wifi", "internet", "online", "digital", "algorithmic",
    # Everyday consumer-tech nouns that actually leak into non-tech captions of
    # tech/screen clips (the highest-frequency real leaks).
    "computer", "laptop", "smartphone", "phone", "screen", "keyboard",
    "email", "glitch", "reboot", "technology", "electronic", "gadget",
]

# Rich style cards: voice + do/don't. Exemplars are appended from FEW_SHOT.
STYLE_CARDS = {
    "formal": (
        "STYLE: Formal\n"
        "Voice: neutral, precise, professional; a broadcast description or serious alt-text.\n"
        "Do: state setting, subject, and key event objectively; consistent tense.\n"
        "Don't: no humor, no irony, no exclamation marks, no editorializing, no slang, no contractions.\n"
        "Trap to avoid: leaking mild amusement. This style is judged on being genuinely straight."
    ),
    "sarcastic": (
        "STYLE: Sarcastic\n"
        "Voice: dry, deadpan, mock-impressed or mock-sympathetic; says the opposite of what it means; understates chaos.\n"
        "Do: irony with a clear target (the situation or decision), faux praise, flat delivery.\n"
        "Don't: no 'haha' energy, no puns for their own sake, no meanness toward identity groups.\n"
        "Trap to avoid: drifting into generic humor -- sarcasm must have an edge and a target."
    ),
    "humorous_tech": (
        "STYLE: Humorous (tech)\n"
        "Voice: comedy built from a software/engineering metaphor mapped onto the event.\n"
        "Do: make the tech concept the MECHANISM of the joke (rollback, race condition, load test, "
        "garbage collection, prod incident, retry loop, edge case, tech debt).\n"
        "Don't: a generic joke with a random tech word bolted on; jargon so deep only SREs laugh; "
        "more than one metaphor per caption.\n"
        "Trap to avoid: becoming non-tech humor wearing the word 'server'."
    ),
    "humorous_non_tech": (
        "STYLE: Humorous (non-tech)\n"
        "Voice: warm observational comedy; family-group-chat energy; relatable exaggeration.\n"
        "Do: mock-drama, universal experiences, playful narration of the payoff.\n"
        "Don't: NO technology references at all; no dry cutting irony (that is the sarcastic lane); no cruelty.\n"
        "Trap to avoid: overlapping with sarcastic -- this style is warm and playful, not dry and cutting."
    ),
}

SYSTEM = (
    "You are an award-winning short-video caption writer. You write captions in a "
    "specified style, grounded strictly in a factual account of the clip. You never "
    "invent details, and you make each style unmistakably distinct from the others."
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
    if gt.get("uncertain"):
        # Never enumerate the uncertain items verbatim: naming them plants the
        # exact nouns we need the writer NOT to use (pink-elephant effect).
        lines.append(
            "Some details (such as exact on-screen text, brands, names, or counts) "
            "could not be verified from the footage — do not mention, quote, or "
            "joke about anything not explicitly listed above."
        )
    conf = gt.get("confidence", 1.0)
    if conf < 0.5:
        lines.append(
            "(NOTE: keep the caption general and grounded; reference only the "
            "details given above and do NOT invent specifics to be funny.)"
        )
    return "\n".join(lines)


def _style_temperature(style_key: str, spec: ModelSpec) -> float:
    return spec.temperature_formal if style_key == "formal" else spec.temperature_humor


def _card_with_examples(style_key: str) -> str:
    card = STYLE_CARDS[style_key]
    examples = FEW_SHOT.get(style_key, [])
    if examples:
        # Show BOTH halves (facts -> caption) so the model learns the
        # accuracy-anchoring habit, and the concrete nouns are clearly attributed
        # to OTHER clips (guards against detail-bleed into the current clip).
        ex_lines = "\n".join(f'FACTS: {gt}\n-> "{cap}"' for gt, cap in examples)
        card += f"\nExamples of this style (from UNRELATED videos):\n{ex_lines}"
    return card


def _build_style_prompt(
    style_key: str, gt: dict[str, Any], comedy_text: str, n: int, critique: str = ""
) -> list[dict[str, Any]]:
    parts = [
        f"Write {n} candidate captions for a short video, in the style defined below. "
        "Each candidate must take a DIFFERENT angle.",
        f"\nFACTS (the only things you may reference):\n{_gt_to_text(gt)}",
    ]
    if style_key in HUMOR_STYLES and comedy_text:
        parts.append(f"\nCOMEDIC MATERIAL (pre-approved, grounded angles):\n{comedy_text}")
    parts.append(f"\nSTYLE CARD:\n{_card_with_examples(style_key)}")

    rules = [
        "ONE tight, punchy sentence per caption, under ~25 words (two sentences only if truly needed).",
        "Every claim must be supported by FACTS above. No invented objects, text, brands, or actions.",
        "Anchor each caption in at least ONE specific, concrete detail from the clip.",
        "The tone must be unmistakable within the first clause.",
        "Output plain caption text: no quotes, no markdown, no 'Caption:' prefixes.",
    ]
    if style_key == "humorous_non_tech":
        rules.append(f"Use NO technology vocabulary at all. Banned words: {', '.join(BANNED_TECH_WORDS)}.")
    parts.append("\nRules:\n- " + "\n- ".join(rules))

    if critique:
        parts.append(f"\nThe previous attempt failed judging. Fix exactly this: {critique}")

    parts.append(f'\nReturn ONLY JSON: {{"candidates": ["<caption 1>", ...]}} with {n} strings.')
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "\n".join(parts)},
    ]


def _parse_candidates(raw: dict[str, Any], n: int) -> list[str]:
    cands = raw.get("candidates")
    if isinstance(cands, str):
        cands = [cands]
    if not isinstance(cands, list):
        # Some models return {"1": "...", "2": "..."} or a bare list under another key.
        vals = [v for v in raw.values() if isinstance(v, str)]
        cands = vals or []
    out = [str(c).strip() for c in cands if str(c).strip()]
    return out[:n] if out else out


def generate_candidates(
    style_key: str,
    gt: dict[str, Any],
    comedy_text: str,
    spec: ModelSpec,
    client: FireworksClient,
    n: int | None = None,
    critique: str = "",
    n_override: int | None = None,
) -> list[str]:
    """Generate N candidate captions for one style."""
    n = n_override or n or spec.n_candidates
    messages = _build_style_prompt(style_key, gt, comedy_text, n, critique=critique)
    temp = _style_temperature(style_key, spec)
    try:
        raw = client.chat_json(spec, messages, temperature=temp)
        cands = _parse_candidates(raw, n)
    except Exception as e:
        log.warning("candidate generation failed for %s: %s", style_key, e)
        cands = []
    return cands


_WRAP_QUOTES = re.compile(r'^\s*["\'“‘](.*)["\'”’]\s*$', re.DOTALL)
_PREFIX = re.compile(r"(?i)^\s*(?:caption|answer|formal|sarcastic|humorous[_ -]?(?:non[_ -]?)?tech)\s*[:\-]\s*")
_THINK = re.compile(r"(?s)<(?:think|thought)>.*?(?:</(?:think|thought)>|\Z)\s*")


def finalize(style_key: str, caption: str) -> str:
    """Deterministic cleanup applied to EVERY winner and fallback: strip
    reasoning blocks, wrapping quotes, label prefixes, markdown fences; collapse
    whitespace; hard-cap runaway length at the second sentence boundary."""
    if not caption:
        return ""
    text = _THINK.sub("", caption)
    text = text.replace("```", " ").strip()
    m = _WRAP_QUOTES.match(text)
    if m:
        text = m.group(1).strip()
    text = _PREFIX.sub("", text)
    text = " ".join(text.split())
    # Cap at two sentences: rambles dilute both judge axes.
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) > 2:
        text = " ".join(sentences[:2])
    return text.strip()


# Stems that are unambiguously technical even as a prefix, so we can match any
# inflection greedily (deploy->deployed/deploying, debug->debugging) with zero
# risk of flagging ordinary English. Every OTHER banned word matches EXACTLY, so
# ambiguous roots never fire falsely: "build"!->"building", "commit"!->"commitment",
# "cloud"!->"clouds", "app"!->"apple", "RAM"!->"ramp".
_SAFE_INFLECT_STEMS = [
    "deploy", "server", "debug", "compile", "upload", "download", "refactor",
    "computer", "laptop", "smartphone", "phone",  # -> computers, phones, ...
]

# Nouns where the plural is exactly as technical as the singular, but a greedy
# \w* would over-match (screen -> screenplay). Allow only s/es.
_PLURAL_ONLY_STEMS = [
    "app", "algorithm", "database", "email", "keyboard", "glitch", "screen",
    "pixel", "browser", "website", "gadget", "robot",
]


def _banned_pattern() -> re.Pattern:
    """One regex over all banned words: greedy inflection for the safe stems,
    exact match for everything else."""
    parts = []
    for w in BANNED_TECH_WORDS:
        esc = re.escape(w.lower())
        if w.lower() in _SAFE_INFLECT_STEMS:
            parts.append(rf"{esc}\w*")
        elif w.lower() in _PLURAL_ONLY_STEMS:
            parts.append(rf"{esc}(?:es|s)?")
        else:
            parts.append(esc)
    return re.compile(rf"\b(?:{'|'.join(parts)})\b", re.IGNORECASE)


_BANNED_RE = _banned_pattern()


def sanitize_non_tech(caption: str) -> tuple[str, bool]:
    """Detect banned tech words (and common inflections) in a non-tech caption.
    Normalizes separators first so 'A.I.' and 'wi-fi' cannot sneak past.
    Returns (caption, clean)."""
    normalized = re.sub(r"[.\-]", "", caption)
    hit = _BANNED_RE.search(caption) or _BANNED_RE.search(normalized)
    return caption, hit is None
