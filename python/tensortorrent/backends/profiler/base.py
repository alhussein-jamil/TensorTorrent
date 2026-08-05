"""Shared profiler types and helpers."""

from __future__ import annotations

import statistics
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

_MAX_TRANSFER_PROFILE_BYTES = 64 << 20


def _bounded_transfer_size(nbytes: int) -> tuple[int, float]:
    # Prefer package attribute so tests can monkeypatch
    # ``tensortorrent.backends.profiler._MAX_TRANSFER_PROFILE_BYTES``.
    try:
        from tensortorrent.backends import profiler as profiler_pkg

        limit = int(profiler_pkg._MAX_TRANSFER_PROFILE_BYTES)
    except Exception:  # pragma: no cover - import cycle / missing attr
        limit = int(_MAX_TRANSFER_PROFILE_BYTES)
    requested = max(0, int(nbytes))
    measured = max(1, min(requested or 1, limit))
    return measured, (float(requested) / float(measured) if requested > measured else 1.0)


@dataclass(frozen=True)
class ProfileRecord:
    device_fingerprint: str
    region_graph_hash: str
    shape: tuple[tuple[int, ...], ...]
    dtype: tuple[str, ...]
    layout: str
    thread_configuration: str
    backend_implementation: str
    warm_up_count: int
    sample_count: int
    median_s: float
    dispersion_s: float
    workspace_memory_bytes: int
    measured: bool
    simulated: bool
    kind: str
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "device_fingerprint": self.device_fingerprint,
            "region_graph_hash": self.region_graph_hash,
            "shape": [list(s) for s in self.shape],
            "dtype": list(self.dtype),
            "layout": self.layout,
            "thread_configuration": self.thread_configuration,
            "backend_implementation": self.backend_implementation,
            "warm_up_count": self.warm_up_count,
            "sample_count": self.sample_count,
            "median_s": self.median_s,
            "dispersion_s": self.dispersion_s,
            "workspace_memory_bytes": self.workspace_memory_bytes,
            "measured": self.measured,
            "simulated": self.simulated,
            "kind": self.kind,
            "notes": list(self.notes),
        }


class BackendProfiler(ABC):
    """Profile regions, transfers, overlap, and memory behavior for one backend."""

    backend_id: str

    @abstractmethod
    def profile_region(
        self,
        fn: Callable[..., Any],
        inputs: tuple[Any, ...],
        *,
        device_fingerprint: str,
        region_graph_hash: str,
        warm_up: int = 2,
        samples: int = 5,
    ) -> ProfileRecord: ...

    @abstractmethod
    def profile_transfer(
        self,
        nbytes: int,
        *,
        source: str,
        destination: str,
        device_fingerprint: str,
        warm_up: int = 1,
        samples: int = 5,
        transfer_fn: Callable[[], None] | None = None,
    ) -> ProfileRecord: ...

    @abstractmethod
    def profile_overlap(
        self,
        compute_fn: Callable[[], None],
        transfer_fn: Callable[[], None],
        *,
        device_fingerprint: str,
        warm_up: int = 1,
        samples: int = 3,
    ) -> ProfileRecord: ...

    @abstractmethod
    def profile_memory_behavior(
        self,
        alloc_fn: Callable[[], Any],
        free_fn: Callable[[Any], None],
        *,
        device_fingerprint: str,
        nbytes: int,
        samples: int = 3,
    ) -> ProfileRecord: ...


def _shapes_dtypes(inputs: tuple[Any, ...]) -> tuple[tuple[tuple[int, ...], ...], tuple[str, ...]]:
    shapes: list[tuple[int, ...]] = []
    dtypes: list[str] = []
    for value in inputs:
        if isinstance(value, torch.Tensor):
            shapes.append(tuple(value.shape))
            dtypes.append(str(value.dtype).replace("torch.", ""))
        else:
            shapes.append(())
            dtypes.append(type(value).__name__)
    return tuple(shapes), tuple(dtypes)


def _summarize(samples: list[float], *, warm_up: int, measured: bool, simulated: bool, **kwargs: Any) -> ProfileRecord:
    ordered = sorted(samples) if samples else [0.0]
    median = statistics.median(ordered)
    dispersion = statistics.pstdev(ordered) if len(ordered) >= 2 else 0.0
    return ProfileRecord(
        device_fingerprint=str(kwargs.get("device_fingerprint", "")),
        region_graph_hash=str(kwargs.get("region_graph_hash", "")),
        shape=tuple(kwargs.get("shape", ())),
        dtype=tuple(kwargs.get("dtype", ())),
        layout=str(kwargs.get("layout", "contiguous")),
        thread_configuration=str(kwargs.get("thread_configuration", f"intraop={torch.get_num_threads()}")),
        backend_implementation=str(kwargs.get("backend_implementation", "")),
        warm_up_count=warm_up,
        sample_count=len(samples),
        median_s=float(median),
        dispersion_s=float(dispersion),
        workspace_memory_bytes=int(kwargs.get("workspace_memory_bytes", 0) or 0),
        measured=measured,
        simulated=simulated,
        kind=str(kwargs.get("kind", "region")),
        notes=tuple(kwargs.get("notes", ()) or ()),
    )
