"""Judging-harness entrypoint for Track 2.

The harness mounts /input/tasks.json as [{task_id, video_url, styles}] and
expects /output/results.json as [{task_id, captions: {style: caption}}] with
the exact style keys from the guide. This module bridges the captioning
pipeline (which works on a local video file) to that contract: it downloads
each clip, runs process_clip, and writes results keyed by task_id with the
requested style keys.

Fail-safe, never silent: a valid results file is written after every clip, and
any clip that errors still appears with its requested style keys (empty), so a
single bad download never zeroes the whole batch.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .cache import Cache
from .client import FireworksClient
from .config import Config
from .pipeline import process_clip
from .styles import STYLES

DEFAULT_STYLES = list(STYLES)
HARD_STOP_MARGIN_S = 20.0  # stop starting new clips this long before the deadline


def run_harness(
    input_path: str = "/input/tasks.json",
    output_path: str = "/output/results.json",
    max_workers: int = 4,   # balance API parallelism vs CPU contention from local Whisper
    download_timeout_s: float = 90.0,
    time_budget_s: float = 570.0,
) -> int:
    started = time.monotonic()
    tasks = _read_tasks(input_path)
    # Seed every task with its requested style keys so output is valid no matter what.
    results: dict[str, dict] = {
        t["task_id"]: {"task_id": t["task_id"], "captions": {s: "" for s in t["styles"]}}
        for t in tasks
    }
    _write(output_path, results)
    if not tasks:
        return 0

    try:
        cfg = Config.load()
    except Exception as exc:
        print(f"config error: {exc}", file=sys.stderr)
        _write(output_path, results)
        return 0
    client = FireworksClient(cfg.api)
    cache = Cache(Path(os.environ.get("CAP_CACHE_DIR", tempfile.gettempdir()) + "/cap_cache"))
    deadline = started + time_budget_s

    def work(task):
        tid, styles, url = task["task_id"], task["styles"], task["video_url"]
        if time.monotonic() >= deadline - HARD_STOP_MARGIN_S:
            return  # out of time: leave the seeded (empty) captions
        tmp = None
        try:
            tmp = _download(url, download_timeout_s)
            res = process_clip(tmp, cfg, client, cache)
            caps = res.get("captions", {}) or {}
            results[tid]["captions"] = {s: (caps.get(s) or "") for s in styles}
        except Exception as exc:
            print(f"clip {tid} failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        finally:
            if tmp is not None:
                try:
                    Path(tmp).unlink()
                except Exception:
                    pass
            _write(output_path, results)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        list(pool.map(work, tasks))

    _write(output_path, results)
    done = sum(1 for r in results.values() if any(r["captions"].values()))
    print(
        json.dumps({"clips": len(tasks), "captioned": done,
                    "elapsed_s": round(time.monotonic() - started, 1)}),
        file=sys.stderr,
    )
    return 0


def _read_tasks(input_path: str) -> list[dict]:
    try:
        raw = json.loads(Path(input_path).read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"cannot read {input_path}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return []
    tasks = []
    for i, item in enumerate(raw if isinstance(raw, list) else []):
        if not isinstance(item, dict):
            continue
        tasks.append({
            "task_id": str(item.get("task_id", f"task-{i}")),
            "styles": list(item.get("styles") or DEFAULT_STYLES),
            "video_url": item.get("video_url", ""),
        })
    return tasks


def _download(url: str, timeout: float) -> Path:
    fd, path = tempfile.mkstemp(suffix=".mp4", prefix="cap_")
    os.close(fd)
    req = urllib.request.Request(url, headers={"User-Agent": "captioner/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(path, "wb") as fh:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
    return Path(path)


def _write(output_path: str, results: dict[str, dict]) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = list(results.values())
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(arr, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


if __name__ == "__main__":
    inp = os.environ.get("INPUT_PATH", "/input/tasks.json")
    out = os.environ.get("OUTPUT_PATH", "/output/results.json")
    raise SystemExit(run_harness(inp, out))
