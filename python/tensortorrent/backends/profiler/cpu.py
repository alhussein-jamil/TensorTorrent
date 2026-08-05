"""CPU backend profiler."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import torch

from tensortorrent.backends.profiler.base import (
    BackendProfiler,
    ProfileRecord,
    _bounded_transfer_size,
    _shapes_dtypes,
    _summarize,
)


class CpuBackendProfiler(BackendProfiler):
    backend_id = "cpu"

    def profile_region(
        self,
        fn: Callable[..., Any],
        inputs: tuple[Any, ...],
        *,
        device_fingerprint: str,
        region_graph_hash: str,
        warm_up: int = 2,
        samples: int = 5,
    ) -> ProfileRecord:
        shapes, dtypes = _shapes_dtypes(inputs)
        for _ in range(max(0, warm_up)):
            fn(*inputs)
        timings: list[float] = []
        for _ in range(max(1, samples)):
            t0 = time.perf_counter()
            fn(*inputs)
            timings.append(time.perf_counter() - t0)
        return _summarize(
            timings,
            warm_up=warm_up,
            measured=True,
            simulated=False,
            device_fingerprint=device_fingerprint,
            region_graph_hash=region_graph_hash,
            shape=shapes,
            dtype=dtypes,
            backend_implementation="cpu",
            kind="region",
        )

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
    ) -> ProfileRecord:
        measured_nbytes, scale = _bounded_transfer_size(nbytes)
        payload = torch.empty(measured_nbytes, dtype=torch.uint8)

        def _default() -> None:
            _ = payload.clone()

        fn = transfer_fn or _default
        for _ in range(max(0, warm_up)):
            fn()
        timings = []
        for _ in range(max(1, samples)):
            t0 = time.perf_counter()
            fn()
            timings.append((time.perf_counter() - t0) * scale)
        return _summarize(
            timings,
            warm_up=warm_up,
            measured=True,
            simulated=False,
            device_fingerprint=device_fingerprint,
            region_graph_hash=f"transfer:{source}->{destination}:{nbytes}",
            shape=((nbytes,),),
            dtype=("uint8",),
            backend_implementation="cpu_memcpy",
            kind="transfer",
            notes=(
                f"source={source}",
                f"destination={destination}",
                f"measured_bytes={measured_nbytes}",
                f"requested_bytes={max(0, int(nbytes))}",
            ),
        )

    def profile_overlap(
        self,
        compute_fn: Callable[[], None],
        transfer_fn: Callable[[], None],
        *,
        device_fingerprint: str,
        warm_up: int = 1,
        samples: int = 3,
    ) -> ProfileRecord:
        from concurrent.futures import ThreadPoolExecutor

        for _ in range(max(0, warm_up)):
            compute_fn()
            transfer_fn()
        timings = []
        with ThreadPoolExecutor(max_workers=2) as pool:
            for _ in range(max(1, samples)):
                t0 = time.perf_counter()
                f1 = pool.submit(compute_fn)
                f2 = pool.submit(transfer_fn)
                f1.result()
                f2.result()
                timings.append(time.perf_counter() - t0)
        return _summarize(
            timings,
            warm_up=warm_up,
            measured=True,
            simulated=False,
            device_fingerprint=device_fingerprint,
            region_graph_hash="overlap:cpu",
            backend_implementation="cpu",
            kind="overlap",
        )

    def profile_memory_behavior(
        self,
        alloc_fn: Callable[[], Any],
        free_fn: Callable[[Any], None],
        *,
        device_fingerprint: str,
        nbytes: int,
        samples: int = 3,
    ) -> ProfileRecord:
        timings = []
        for _ in range(max(1, samples)):
            t0 = time.perf_counter()
            handle = alloc_fn()
            free_fn(handle)
            timings.append(time.perf_counter() - t0)
        return _summarize(
            timings,
            warm_up=0,
            measured=True,
            simulated=False,
            device_fingerprint=device_fingerprint,
            region_graph_hash=f"memory:{nbytes}",
            backend_implementation="cpu",
            kind="memory",
            workspace_memory_bytes=nbytes,
        )
