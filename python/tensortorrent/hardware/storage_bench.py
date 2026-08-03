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

    def as_dict(self) -> dict[str, float | int | str | bool]:
        return {
            "path": self.path,
            "nbytes": self.nbytes,
            "latency_s": self.latency_s,
            "bytes_per_s": self.bytes_per_s,
            "measured": self.measured,
            "notes": self.notes,
        }


def benchmark_sequential_read(path: Path, nbytes: int = 16 << 20) -> StorageBenchResult:
    path = Path(path)
    if path.is_dir():
        target = path / ".tensortorrent_read_probe"
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
        return benchmark_pread(target, offset=0, nbytes=nbytes, notes="sequential pread")
    finally:
        if cleanup:
            import contextlib

            with contextlib.suppress(OSError):
                target.unlink()


def benchmark_pread(
    path: Path,
    *,
    offset: int,
    nbytes: int,
    iters: int = 3,
    notes: str = "pread",
) -> StorageBenchResult:
    """Time real ``os.pread`` calls — the same syscall the streaming store uses."""
    path = Path(path)
    if nbytes <= 0:
        return StorageBenchResult(str(path), 0, 0.0, 0.0, False, "empty read")
    try:
        file_size = path.stat().st_size
    except OSError as exc:
        return StorageBenchResult(str(path), nbytes, float("inf"), 0.0, False, str(exc))
    if offset < 0 or offset + nbytes > file_size:
        return StorageBenchResult(
            str(path),
            nbytes,
            float("inf"),
            0.0,
            False,
            f"range [{offset}, {offset + nbytes}) outside file size {file_size}",
        )
    fd = os.open(path, os.O_RDONLY)
    try:
        warm = os.pread(fd, nbytes, offset)
        if len(warm) != nbytes:
            return StorageBenchResult(str(path), nbytes, float("inf"), 0.0, False, "short warmup read")
        start = time.perf_counter()
        for _ in range(max(1, iters)):
            chunk = os.pread(fd, nbytes, offset)
            if len(chunk) != nbytes:
                return StorageBenchResult(str(path), nbytes, float("inf"), 0.0, False, "short timed read")
        elapsed = (time.perf_counter() - start) / max(1, iters)
        bps = nbytes / elapsed if elapsed > 0 else 0.0
        return StorageBenchResult(str(path), nbytes, elapsed, bps, True, notes)
    finally:
        os.close(fd)


def benchmark_pack_payload(path: Path, *, offset: int, nbytes: int) -> StorageBenchResult:
    """Measure payload bandwidth for one model-pack block."""
    return benchmark_pread(
        path,
        offset=offset,
        nbytes=nbytes,
        iters=3,
        notes="model pack payload pread",
    )


def benchmark_storage_resources(mountpoints: list[str]) -> list[StorageBenchResult]:
    out: list[StorageBenchResult] = []
    for mp in mountpoints:
        try:
            out.append(benchmark_sequential_read(Path(mp)))
        except OSError as exc:
            out.append(StorageBenchResult(mp, 0, float("inf"), 0.0, False, str(exc)))
    return out
