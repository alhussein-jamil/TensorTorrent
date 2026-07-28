"""Storage and NVMe transfer benchmarking."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class StorageBenchResult:
    path: str
    nbytes: int
    latency_s: float
    bytes_per_s: float
    measured: bool
    notes: str = ""


def benchmark_sequential_read(path: Path, nbytes: int = 16 << 20) -> StorageBenchResult:
    path = Path(path)
    if path.is_dir():
        target = path / ".streamcompiler_read_probe"
        data = os.urandom(nbytes)
        target.write_bytes(data)
        cleanup = True
    else:
        target = path
        cleanup = False
        if not target.exists():
            return StorageBenchResult(str(path), nbytes, float("inf"), 0.0, False, "missing path")
        nbytes = min(nbytes, target.stat().st_size)
    try:
        # Cold-ish read: drop page cache is privileged; measure what we can.
        with open(target, "rb") as fh:
            fh.read(1)
            fh.seek(0)
            start = time.perf_counter()
            remaining = nbytes
            while remaining > 0:
                chunk = fh.read(min(1 << 20, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
            elapsed = time.perf_counter() - start
        read_bytes = nbytes - remaining
        bps = read_bytes / elapsed if elapsed > 0 else 0.0
        return StorageBenchResult(str(target), read_bytes, elapsed, bps, True, "sequential read")
    finally:
        if cleanup:
            import contextlib

            with contextlib.suppress(OSError):
                target.unlink()


def benchmark_storage_resources(mountpoints: list[str]) -> list[StorageBenchResult]:
    out: list[StorageBenchResult] = []
    for mp in mountpoints:
        try:
            out.append(benchmark_sequential_read(Path(mp)))
        except OSError as exc:
            out.append(StorageBenchResult(mp, 0, float("inf"), 0.0, False, str(exc)))
    return out
