"""Batch CLI entry point.

Usage:
    python -m captioner.cli --input ./clips --output ./output/captions.json

Walks a directory of clips, processes each with per-clip error isolation, and
writes one JSON file with the four captions per clip. Designed to be the Docker
ENTRYPOINT.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from tqdm import tqdm

from .cache import Cache
from .client import FireworksClient
from .config import Config
from .pipeline import process_clip
from .preprocess import VIDEO_EXTS


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


def _find_clips(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    clips = sorted(p for p in input_path.rglob("*") if p.suffix.lower() in VIDEO_EXTS)
    return clips


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Four-style video captioner (Track 2).")
    parser.add_argument("--input", "-i", required=True, help="Video file or directory of clips.")
    parser.add_argument("--output", "-o", default="output/captions.json", help="Output JSON path.")
    parser.add_argument("--config", "-c", default=None, help="Path to models.yaml (default: config/models.yaml).")
    parser.add_argument("--cache-dir", default="cache", help="Stage A cache directory.")
    parser.add_argument("--max-frames", type=int, default=None, help="Override max frames per clip.")
    parser.add_argument("--no-judge", action="store_true", help="Disable the self-critique loop.")
    parser.add_argument("--keep-work", action="store_true", help="Keep temp frames/audio for debugging.")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N clips.")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    _setup_logging(args.verbose)
    log = logging.getLogger("captioner.cli")

    try:
        cfg = Config.load(args.config)
    except Exception as e:
        log.error("config error: %s", e)
        return 2

    input_path = Path(args.input)
    if not input_path.exists():
        log.error("input path does not exist: %s", input_path)
        return 2

    clips = _find_clips(input_path)
    if args.limit:
        clips = clips[: args.limit]
    if not clips:
        log.error("no video files found under %s (looked for %s)", input_path, sorted(VIDEO_EXTS))
        return 2
    log.info("found %d clip(s)", len(clips))

    client = FireworksClient(cfg.api)
    cache = Cache(Path(args.cache_dir))

    results = []
    n_ok = n_err = 0
    for clip in tqdm(clips, desc="captioning", unit="clip"):
        res = process_clip(
            clip, cfg, client, cache,
            max_frames=args.max_frames,
            keep_work=args.keep_work,
            run_judge=(False if args.no_judge else None),
        )
        if res.get("error"):
            n_err += 1
            log.warning("%s -> ERROR: %s", clip.name, res["error"])
        else:
            n_ok += 1
        results.append(res)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "count": len(results),
        "ok": n_ok,
        "errors": n_err,
        "results": results,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    log.info("wrote %s (%d ok, %d errors)", out_path, n_ok, n_err)
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
