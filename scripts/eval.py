"""Offline eval loop: generate -> judge-select -> score sheet.

Run over your own diverse test clips to iterate on prompts BEFORE the real judge
sees you. Prints a per-clip x per-style grid of accuracy/tone (the "score sheet"
from the playbook) so you can see exactly which cells are weak and which prompt
to tune next.

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

    labels = [STYLE_LABELS[s] for s in STYLES]
    per_style = {lbl: {"acc": [], "tone": []} for lbl in labels}
    grid_rows = []   # (clip, {label: (acc, tone)})
    weak = []
    rows = []

    for clip in clips:
        res = process_clip(clip, cfg, client, cache, run_judge=True)
        if res.get("error"):
            print(f"  x {clip.name}: {res['error']}")
            continue
        sel = res.get("selection", {})
        cell = {}
        clip_scores = []
        for lbl in labels:
            s = sel.get(lbl, {})
            acc, tone = s.get("accuracy", 0), s.get("tone", 0)
            per_style[lbl]["acc"].append(acc)
            per_style[lbl]["tone"].append(tone)
            cell[lbl] = (acc, tone)
            clip_scores.append(min(acc, tone))
        grid_rows.append((clip.name, cell))
        worst = min(clip_scores) if clip_scores else 0
        print(f"  ok {clip.name}: worst-cell={worst}")
        if worst < cfg.critique.min_score:
            weak.append({"file": clip.name, "worst": worst, "captions": res["captions"], "selection": sel})
        rows.append(res)

    def avg(xs):
        return round(sum(xs) / len(xs), 2) if xs else 0.0

    # Score-sheet grid: rows = clips, cols = styles, cell = "acc/tone".
    print("\n=== SCORE SHEET (accuracy/tone) ===")
    header = "clip".ljust(28) + "".join(l[:12].ljust(14) for l in labels)
    print(header)
    for name, cell in grid_rows:
        line = name[:27].ljust(28)
        for lbl in labels:
            a, t = cell[lbl]
            line += f"{a:.0f}/{t:.0f}".ljust(14)
        print(line)
    print("-" * len(header))
    avg_line = "AVG".ljust(28)
    for lbl in labels:
        avg_line += f"{avg(per_style[lbl]['acc']):.1f}/{avg(per_style[lbl]['tone']):.1f}".ljust(14)
    print(avg_line)

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
