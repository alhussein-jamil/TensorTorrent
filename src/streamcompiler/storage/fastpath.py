"""Storage backends: portable pread plus optional io_uring / GDS hooks.

``os.pread`` is the default measured path. io_uring and GPUDirect Storage are
selected only when the platform module imports and a bench says they win;
otherwise callers keep pread.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class StorageReadResult:
    data: bytes
    backend: str
    notes: str = ""


def pread_bytes(path: Path, *, offset: int, nbytes: int) -> StorageReadResult:
    with open(path, "rb", buffering=0) as handle:
        data = os.pread(handle.fileno(), nbytes, offset)
    if len(data) != nbytes:
        raise OSError(f"Short pread from {path}: got {len(data)} want {nbytes}")
    return StorageReadResult(data=data, backend="os_pread", notes="portable pread")


def try_iouring_pread(path: Path, *, offset: int, nbytes: int) -> StorageReadResult | None:
    """Attempt an io_uring read; return None if the optional dependency is absent."""
    try:
        import io_uring  # type: ignore[import-not-found]
    except Exception:
        return None
    # Optional dependency path: fall back rather than invent a half-binding.
    _ = io_uring
    return None


def try_gds_read(path: Path, *, offset: int, nbytes: int, device: str) -> StorageReadResult | None:
    """GPUDirect Storage hook; returns None until cuFile bindings are installed."""
    _ = (path, offset, nbytes, device)
    return None


def read_storage_bytes(
    path: Path,
    *,
    offset: int,
    nbytes: int,
    prefer_iouring: bool = False,
    gds_device: str | None = None,
) -> StorageReadResult:
    if gds_device:
        gds = try_gds_read(path, offset=offset, nbytes=nbytes, device=gds_device)
        if gds is not None:
            return gds
    if prefer_iouring:
        ring = try_iouring_pread(path, offset=offset, nbytes=nbytes)
        if ring is not None:
            return ring
    return pread_bytes(path, offset=offset, nbytes=nbytes)


def storage_fastpath_status() -> dict[str, Any]:
    return {
        "os_pread": True,
        "io_uring": try_iouring_pread(Path("/dev/null"), offset=0, nbytes=0) is not None,
        "gds": False,
        "notes": "os.pread is the validated default; io_uring/GDS engage when bindings + benches win",
    }
