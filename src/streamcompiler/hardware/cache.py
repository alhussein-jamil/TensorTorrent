"""Benchmark cache keyed by machine fingerprint."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from streamcompiler.hardware.fingerprint import machine_fingerprint

logger = logging.getLogger(__name__)

_MAX_CACHE_BYTES = 256 * 1024 * 1024


class BenchmarkCache:
    def __init__(self, cache_dir: Path, fingerprint: str | None = None) -> None:
        self.cache_dir = Path(cache_dir)
        self.fingerprint = fingerprint or machine_fingerprint()
        self.root = self.cache_dir / self.fingerprint
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, category: str, key: str) -> Path:
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in key)
        return self.root / category / f"{safe}.json"

    def get(self, category: str, key: str) -> dict[str, Any] | None:
        path = self.path_for(category, key)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("cache read failed for %s: %s", path, exc)
            return None

    def put(self, category: str, key: str, payload: dict[str, Any]) -> None:
        path = self.path_for(category, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
        self._enforce_bound()

    def _enforce_bound(self) -> None:
        files = sorted(self.root.rglob("*.json"), key=lambda p: p.stat().st_mtime)
        total = sum(p.stat().st_size for p in files)
        import contextlib

        while total > _MAX_CACHE_BYTES and files:
            victim = files.pop(0)
            total -= victim.stat().st_size
            with contextlib.suppress(OSError):
                victim.unlink()
