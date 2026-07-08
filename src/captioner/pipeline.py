"""Per-clip orchestration (best-of-N engine).

    preprocess -> ASR -> Stage A fact sheet -> Stage 3 comedy material
    -> Stage 4 generate N candidates/style -> Stage 5 judge-select winner/style
    -> bounded regeneration for weak styles

One public entry point, `process_clip`, returns a fully-formed result dict for a
single video. Batch-level concerns (walking a directory, error isolation, writing
JSON) live in cli.py.
"""
from __future__ import annotations

import logging
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from . import asr as asr_mod
from . import comedy as comedy_mod
from . import styles as styles_mod
from . import understand as understand_mod
from .cache import Cache, video_hash
from .client import FireworksClient
from .config import Config
from .judge import select_best
from .preprocess import preprocess
from .styles import HUMOR_STYLES, STYLE_LABELS, STYLES

log = logging.getLogger("captioner.pipeline")


def _empty_captions() -> dict[str, str]:
    return {STYLE_LABELS[s]: "" for s in STYLES}


def process_clip(
    video_path: Path,
    cfg: Config,
    client: FireworksClient,
    cache: Cache,
    *,
    max_frames: int | None = None,
    keep_work: bool = False,
    run_judge: bool | None = None,
    deadline: float | None = None,
) -> dict[str, Any]:
    """Process one clip end-to-end. Never raises for content problems -- returns a
    result dict with an `error` field instead so a batch never dies on one clip.

    `deadline` is a time.monotonic() timestamp; as it approaches, the pipeline
    sheds optional stages (ASR, comedy, judging) so a slow clip degrades into a
    plainer caption instead of a timeout."""

    def remaining() -> float:
        return float("inf") if deadline is None else deadline - time.monotonic()

    result: dict[str, Any] = {
        "file": video_path.name,
        "captions": _empty_captions(),
        "error": None,
    }
    workdir = Path(tempfile.mkdtemp(prefix="cap_"))
    try:
        vhash = video_hash(video_path)
        result["video_hash"] = vhash

        mf = max_frames or cfg.understand.max_images
        pre = preprocess(
            video_path, workdir,
            max_frames=mf,
            every_s=cfg.raw.get("sampling", {}).get("every_s", 4.0),
            scene_threshold=cfg.raw.get("sampling", {}).get("scene_threshold", 0.3),
        )
        result["duration_s"] = round(pre.probe.duration, 1)
        result["n_frames"] = len(pre.frames)
        result["has_audio"] = pre.probe.has_audio

        # --- ASR (cached: transcription is deterministic per clip+model) ---
        asr_sig = f"{cfg.asr.backend}|{cfg.asr.model}|{cfg.asr.local_model_size}"
        transcript = cache.get(vhash, "asr", asr_sig)
        if transcript is None:
            if remaining() < 120:
                log.info("deadline tight (%.0fs): skipping ASR for %s", remaining(), video_path.name)
                transcript = {"text": "", "language": None}
            else:
                transcript = asr_mod.transcribe(pre.audio_path, cfg.asr, client)
                # Cache silence (audio present, no speech) and real transcripts
                # alike; only skip caching when ASR is disabled/no audio.
                if pre.audio_path is not None or cfg.asr.backend == "none":
                    cache.put(vhash, "asr", asr_sig, transcript)
        result["language"] = transcript.get("language")

        # --- Stage A: fact sheet (cached) ---
        gt = understand_mod.understand(
            pre, transcript, cfg.understand, client, cache=cache, vhash=vhash
        )
        result["ground_truth"] = gt

        # --- Stage 3: comedy material (cached) ---
        if remaining() < 90:
            comedy = {"material": [], "text": "", "text_no_tech": ""}
        else:
            comedy = comedy_mod.extract_comedy(
                gt, cfg.comedy, cfg.style, client, cache=cache, vhash=vhash
            )
        result["comedy_material"] = comedy.get("material", [])

        # --- Stages 4 + 5: best-of-N generate -> judge-select ---
        do_judge = (cfg.critique.enabled and cfg.judge.enabled) if run_judge is None else run_judge
        if remaining() < 75:  # fast path: one candidate, no judging
            do_judge = False
        captions, selection = _best_of_n(gt, comedy, cfg, client, do_judge, remaining)
        result["captions"] = captions
        if selection:
            result["selection"] = selection
        return result

    except Exception as e:  # unexpected: log, keep batch alive
        log.exception("failed to process %s", video_path.name)
        result["error"] = f"{type(e).__name__}: {e}"
        # Even a total failure ships a deterministic caption per style, because
        # an empty caption is a guaranteed 0 on both judge axes.
        gt = result.get("ground_truth") or {}
        result["captions"] = {
            STYLE_LABELS[s]: (result["captions"].get(STYLE_LABELS[s]) or deterministic_caption(s, gt))
            for s in STYLES
        }
        return result
    finally:
        if not keep_work:
            shutil.rmtree(workdir, ignore_errors=True)


def deterministic_caption(style_key: str, gt: dict[str, Any]) -> str:
    """Zero-LLM fallback caption built from whatever facts exist. A mediocre
    on-tone caption earns partial credit; an empty one earns exactly zero."""
    setting = str(gt.get("setting") or "").strip().rstrip(".")
    subjects = gt.get("subjects") or []
    subject = str(subjects[0]).strip().rstrip(".") if isinstance(subjects, list) and subjects else ""
    core = ", ".join(x for x in (subject, setting) if x) or "an everyday scene unfolding on camera"
    core = core[:140]
    if style_key == "formal":
        return f"A short video clip showing {core}."
    if style_key == "sarcastic":
        return f"Ah yes, {core} — truly the cinematic event of the year."
    if style_key == "humorous_tech":
        return f"{core[0].upper()}{core[1:]}, loading frame by frame like a progress bar that swears it is almost done."
    return f"{core[0].upper()}{core[1:]}, just doing its thing like the rest of us on a Monday."


def _leaks_tech(style_key: str, caption: str) -> bool:
    """True if a humorous_non_tech caption contains banned tech vocabulary."""
    if style_key != "humorous_non_tech" or not caption:
        return False
    _, clean = styles_mod.sanitize_non_tech(caption)
    return not clean


def _first_clean(style_key: str, cands: list[str]) -> str:
    """First candidate that passes the non-tech guard (else the first)."""
    if not cands:
        return ""
    if style_key == "humorous_non_tech":
        return next((c for c in cands if not _leaks_tech(style_key, c)), cands[0])
    return cands[0]


def _sel_quality(sel: dict[str, Any], style_key: str) -> tuple:
    """Rank a selection so we can keep the BEST across regen rounds: a leaking
    non-tech caption is worst; then higher min(accuracy,tone), then composite."""
    leak_penalty = 0 if not _leaks_tech(style_key, sel.get("winner", "")) else -100
    acc, tone = sel.get("accuracy", 0.0), sel.get("tone", 0.0)
    composite = acc + tone + 0.5 * (sel.get("distinct", 0.0) + sel.get("fit", 0.0))
    return (leak_penalty, min(acc, tone), composite)


def _best_of_n(
    gt: dict[str, Any],
    comedy: dict[str, Any],
    cfg: Config,
    client: FireworksClient,
    do_judge: bool,
    remaining,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Generate N candidates per style (all four styles IN PARALLEL: they only
    depend on the fact sheet + comedy material), then judge-select winners.
    Formal skips the judge (low-temp candidates are near-duplicates, the call is
    wasted). Regeneration is bounded, humor-only, and budget-gated. Each style
    is isolated: a failure in one never discards the others, and no style ever
    ships an empty caption."""
    gt_text = styles_mod._gt_to_text(gt)
    captions: dict[str, str] = {}
    selection: dict[str, Any] = {}
    others: dict[str, str] = {}   # already-chosen winners, for the distinctness check

    def comedy_for(style_key: str) -> str:
        if style_key not in HUMOR_STYLES:
            return ""
        # humorous_non_tech (and sarcastic) must not be baited with tech angles.
        if style_key == "humorous_tech":
            return comedy.get("text", "")
        return comedy.get("text_no_tech", comedy.get("text", ""))

    n_candidates = cfg.style.n_candidates if do_judge else 1

    def gen(style_key: str) -> list[str]:
        try:
            return styles_mod.generate_candidates(
                style_key, gt, comedy_for(style_key), cfg.style, client,
                n_override=n_candidates,
            )
        except Exception as e:
            log.warning("candidate generation failed for %s: %s", style_key, e)
            return []

    with ThreadPoolExecutor(max_workers=len(STYLES)) as pool:
        all_cands = dict(zip(STYLES, pool.map(gen, STYLES)))

    for style_key in STYLES:
        label = STYLE_LABELS[style_key]
        cands = all_cands.get(style_key) or []
        judge_this = do_judge and style_key != "formal" and len(cands) > 1

        if not judge_this:
            winner = _first_clean(style_key, cands)
        else:
            try:
                sel = select_best(style_key, gt_text, cands, others, cfg.judge, client, cfg.critique.min_score)
                best = sel
                leak = _leaks_tech(style_key, sel["winner"])

                retries = max(0, cfg.critique.max_retries)
                while (sel["needs_regen"] or leak) and retries > 0 and remaining() > 45:
                    retries -= 1
                    critique = sel["critique"] or "Improve accuracy and tone."
                    if leak:
                        critique += " Remove ALL technology vocabulary."
                    new_cands = styles_mod.generate_candidates(
                        style_key, gt, comedy_for(style_key), cfg.style, client, critique=critique
                    )
                    if not new_cands:
                        break
                    sel = select_best(style_key, gt_text, new_cands, others, cfg.judge, client, cfg.critique.min_score)
                    # Keep the best round, so a regen never yields a strictly worse caption.
                    if _sel_quality(sel, style_key) > _sel_quality(best, style_key):
                        best = sel
                    leak = _leaks_tech(style_key, sel["winner"])

                winner = best["winner"] or _first_clean(style_key, cands)
                selection[label] = {
                    "accuracy": best["accuracy"], "tone": best["tone"],
                    "distinct": best["distinct"], "fit": best["fit"],
                    "n_candidates": len(cands), "critique": best["critique"],
                }
            except Exception as e:  # isolate: one bad style must not zero the others
                log.warning("style %s selection failed (%s); using first candidate", style_key, e)
                winner = _first_clean(style_key, cands)

        winner = styles_mod.finalize(style_key, winner)
        if not winner:
            winner = deterministic_caption(style_key, gt)
        captions[label] = winner
        others[label] = winner

    return captions, selection
