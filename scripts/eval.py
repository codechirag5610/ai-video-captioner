"""Offline eval loop: generate -> judge -> aggregate.

Run over your own diverse test clips to iterate on prompts BEFORE the real judge
sees you. Prints per-style average accuracy/tone and flags weak clips so you know
exactly which prompt to tune next.

    python scripts/eval.py --input ./clips --out output/eval.json

Tip: collect ~10-15 clips mirroring the likely test distribution (animal fails,
screen recordings, sports bloopers, talking heads, silent b-roll, memes).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from captioner.cache import Cache          # noqa: E402
from captioner.client import FireworksClient  # noqa: E402
from captioner.config import Config        # noqa: E402
from captioner.pipeline import process_clip  # noqa: E402
from captioner.preprocess import VIDEO_EXTS  # noqa: E402
from captioner.styles import STYLE_LABELS, STYLES  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", "-i", default="clips")
    ap.add_argument("--out", "-o", default="output/eval.json")
    ap.add_argument("--config", "-c", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = Config.load(args.config)
    client = FireworksClient(cfg.api)
    cache = Cache(Path("cache"))

    clips = sorted(p for p in Path(args.input).rglob("*") if p.suffix.lower() in VIDEO_EXTS)
    if args.limit:
        clips = clips[: args.limit]
    if not clips:
        print(f"No clips found under {args.input}")
        return 1

    per_style = {STYLE_LABELS[s]: {"acc": [], "tone": []} for s in STYLES}
    weak = []
    rows = []

    for clip in clips:
        res = process_clip(clip, cfg, client, cache, run_judge=True)
        if res.get("error"):
            print(f"  ✗ {clip.name}: {res['error']}")
            continue
        judged = res.get("judge") or {}
        scores = judged.get("scores", {})
        for label, sc in scores.items():
            if label in per_style:
                per_style[label]["acc"].append(sc.get("accuracy", 0))
                per_style[label]["tone"].append(sc.get("tone", 0))
        overall = judged.get("overall", 0)
        print(f"  ✓ {clip.name}: overall={overall}  distinguishable={judged.get('distinguishable')}")
        if overall < cfg.critique.min_score:
            weak.append({"file": clip.name, "overall": overall, "captions": res["captions"], "judge": judged})
        rows.append(res)

    def avg(xs):
        return round(sum(xs) / len(xs), 2) if xs else 0.0

    print("\n=== AVERAGE SCORES BY STYLE ===")
    for label, d in per_style.items():
        print(f"  {label:20s}  accuracy={avg(d['acc']):5}  tone={avg(d['tone']):5}")

    summary = {
        "n_clips": len(rows),
        "per_style_avg": {k: {"accuracy": avg(v["acc"]), "tone": avg(v["tone"])} for k, v in per_style.items()},
        "weak_clips": weak,
        "results": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nWrote {out}. {len(weak)} weak clip(s) flagged for prompt tuning.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
