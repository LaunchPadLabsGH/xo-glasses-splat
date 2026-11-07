"""Per-stage manifests: what ran, with which parameters and inputs, how long it took.

Each stage writes `<run>/<stage>/manifest.json`. A stage is skipped when a
manifest exists whose parameter hash and upstream manifest hashes match, so
re-running the pipeline only redoes what changed. The manifests double as the
reproducibility record for TRL 4 evidence.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MANIFEST = "manifest.json"


def stable_hash(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def file_hash(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()[:16]


def read_manifest(stage_dir: Path) -> dict | None:
    p = Path(stage_dir) / MANIFEST
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def upstream_key(run_dir: Path, stages: list[str]) -> dict[str, str | None]:
    """Fingerprint of upstream stages: a change upstream invalidates downstream caches."""
    out: dict[str, str | None] = {}
    for s in stages:
        m = read_manifest(Path(run_dir) / s)
        out[s] = m["fingerprint"] if m else None
    return out


def versions() -> dict[str, str]:
    v = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ("numpy", "cv2", "pycolmap", "torch", "gsplat"):
        try:
            v[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001 - optional modules
            pass
    return v


def code_hash(modules: list[str]) -> str:
    """Hash of the source files implementing a stage, so code changes invalidate caches."""
    import importlib

    h = hashlib.sha256()
    for name in sorted(modules):
        mod = importlib.import_module(name)
        h.update(Path(mod.__file__).read_bytes())
    return h.hexdigest()[:16]


class Stage:
    """Context for one pipeline stage."""

    def __init__(self, run_dir: Path, name: str, params: dict, upstream: list[str] | None = None,
                 code: list[str] | None = None):
        self.run_dir = Path(run_dir)
        self.name = name
        self.dir = self.run_dir / name
        self.params = params
        self.upstream = upstream or []
        self.metrics: dict[str, Any] = {}
        self.outputs: list[str] = []
        self.code = code_hash(code) if code else None
        self.fingerprint = stable_hash({"params": params, "upstream": upstream_key(run_dir, self.upstream),
                                        "code": self.code})

    def is_cached(self) -> bool:
        m = read_manifest(self.dir)
        return bool(m and m.get("fingerprint") == self.fingerprint and m.get("status") == "ok")

    def cached_metrics(self) -> dict:
        m = read_manifest(self.dir) or {}
        return m.get("metrics", {})

    @contextmanager
    def running(self) -> Iterator[Stage]:
        self.dir.mkdir(parents=True, exist_ok=True)
        started = time.time()
        status = "error"
        try:
            yield self
            status = "ok"
        finally:
            manifest = {
                "stage": self.name,
                "status": status,
                "fingerprint": self.fingerprint,
                "params": self.params,
                "code_hash": self.code,
                "upstream": upstream_key(self.run_dir, self.upstream),
                "started_utc": datetime.fromtimestamp(started, timezone.utc).isoformat(),
                "duration_s": round(time.time() - started, 3),
                "metrics": self.metrics,
                "outputs": self.outputs,
                "versions": versions(),
            }
            with open(self.dir / MANIFEST, "w") as f:
                json.dump(manifest, f, indent=2, default=str)
