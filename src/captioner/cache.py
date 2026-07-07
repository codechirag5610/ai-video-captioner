"""Content-addressed cache for expensive Stage A understanding.

Keyed by (video content hash + understand model + sampling params). Iterating on
Stage B style prompts never re-pays for vision inference -> protects credits.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger("captioner.cache")


def video_hash(video_path: Path, chunk_mb: int = 8) -> str:
    """Hash of file size + head/tail bytes. Fast and collision-safe enough for
    a fixed clip set; avoids reading multi-hundred-MB files in full."""
    h = hashlib.sha256()
    size = video_path.stat().st_size
    h.update(str(size).encode())
    chunk = chunk_mb * 1024 * 1024
    with open(video_path, "rb") as f:
        h.update(f.read(chunk))
        if size > chunk:
            f.seek(max(0, size - chunk))
            h.update(f.read(chunk))
    return h.hexdigest()[:16]


class Cache:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _key(self, vhash: str, namespace: str, sig: str) -> str:
        digest = hashlib.sha256(sig.encode()).hexdigest()[:8]
        return f"{namespace}_{vhash}_{digest}"

    def get(self, vhash: str, namespace: str, sig: str) -> dict[str, Any] | None:
        p = self.root / f"{self._key(vhash, namespace, sig)}.json"
        if p.exists():
            try:
                return json.loads(p.read_text())
            except Exception as e:
                log.warning("corrupt cache entry %s: %s", p, e)
        return None

    def put(self, vhash: str, namespace: str, sig: str, value: dict[str, Any]) -> None:
        p = self.root / f"{self._key(vhash, namespace, sig)}.json"
        p.write_text(json.dumps(value, ensure_ascii=False, indent=2))
