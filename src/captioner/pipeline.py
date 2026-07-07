"""Per-clip orchestration: preprocess -> understand -> style -> (critique loop).

One public entry point, `process_clip`, returns a fully-formed result dict for a
single video. Batch-level concerns (walking a directory, error isolation, writing
JSON) live in cli.py.
"""
from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any

from . import asr as asr_mod
from . import styles as styles_mod
from . import understand as understand_mod
from .cache import Cache, video_hash
from .client import FireworksClient
from .config import Config
from .judge import judge as judge_fn
from .preprocess import preprocess
from .styles import STYLE_LABELS, STYLES

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
) -> dict[str, Any]:
    """Process one clip end-to-end. Never raises for content problems — returns a
    result dict with an `error` field instead so a batch never dies on one clip."""
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

        # --- ASR ---
        transcript = asr_mod.transcribe(pre.audio_path, cfg.asr, client)
        result["language"] = transcript.get("language")

        # --- Stage A: understanding (cached) ---
        gt = understand_mod.understand(
            pre, transcript, cfg.understand, client, cache=cache, vhash=vhash
        )
        result["ground_truth"] = gt

        # --- Stage B: styles ---
        captions = styles_mod.generate(gt, cfg.style, client)

        # --- Self-critique loop ---
        do_judge = cfg.critique.enabled if run_judge is None else run_judge
        if do_judge:
            captions, judged = _critique_loop(gt, captions, cfg, client)
            result["judge"] = judged

        result["captions"] = captions
        return result

    except Exception as e:  # unexpected: log, keep batch alive
        log.exception("failed to process %s", video_path.name)
        result["error"] = f"{type(e).__name__}: {e}"
        return result
    finally:
        if not keep_work:
            shutil.rmtree(workdir, ignore_errors=True)


def _critique_loop(
    gt: dict[str, Any],
    captions: dict[str, str],
    cfg: Config,
    client: FireworksClient,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Judge, then regenerate any style below threshold (bounded retries)."""
    gt_text = styles_mod._gt_to_text(gt)
    judged = judge_fn(gt_text, captions, cfg.judge, client)

    for _ in range(max(0, cfg.critique.max_retries)):
        to_fix: list[str] = []
        for s in STYLES:
            label = STYLE_LABELS[s]
            sc = judged["scores"].get(label, {})
            below = min(sc.get("accuracy", 10), sc.get("tone", 10)) < cfg.critique.min_score
            leak = (s == "humorous_non_tech" and judged.get("nontech_has_tech_words"))
            blur = (s in ("humorous_tech", "humorous_non_tech") and not judged.get("distinguishable"))
            if below or leak or blur:
                to_fix.append(s)
        if not to_fix:
            break

        for s in to_fix:
            label = STYLE_LABELS[s]
            critique = judged["scores"].get(label, {}).get("critique", "Improve accuracy and tone.")
            if s == "humorous_non_tech" and judged.get("nontech_has_tech_words"):
                critique += " Remove ALL technology vocabulary."
            if s in ("humorous_tech", "humorous_non_tech") and not judged.get("distinguishable"):
                critique += " Make it clearly distinct from the other humorous style."
            try:
                new_cap = styles_mod.regenerate_one(gt, s, critique, cfg.style, client)
                if new_cap:
                    captions[label] = new_cap
            except Exception as e:
                log.warning("regen failed for %s: %s", s, e)

        judged = judge_fn(gt_text, captions, cfg.judge, client)

    return captions, judged
