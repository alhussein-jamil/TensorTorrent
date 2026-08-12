"""Virtual accelerator profiler."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from tensortorrent.backends.profiler.base import BackendProfiler, ProfileRecord, _shapes_dtypes, _summarize
from tensortorrent.closed import ProfileKind


class VirtualAccelBackendProfiler(BackendProfiler):
    """Deterministic virtual-accelerator profiler — always labelled simulated."""

    backend_id = "mock_accel"

    def __init__(self, *, compute_delay_s: float = 0.05, transfer_delay_s: float = 0.08) -> None:
        self.compute_delay_s = float(compute_delay_s)
        self.transfer_delay_s = float(transfer_delay_s)

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
        # Execute once for correctness; report configured virtual delay as the cost.
        if warm_up > 0:
            fn(*inputs)
        timings = [self.compute_delay_s for _ in range(max(1, samples))]
        return _summarize(
            timings,
            warm_up=warm_up,
            measured=False,
            simulated=True,
            device_fingerprint=device_fingerprint,
            region_graph_hash=region_graph_hash,
            shape=shapes,
            dtype=dtypes,
            backend_implementation="mock_accel",
            kind=ProfileKind.REGION,
            notes=("simulated virtual accelerator",),
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
        timings = [self.transfer_delay_s for _ in range(max(1, samples))]
        return _summarize(
            timings,
            warm_up=warm_up,
            measured=False,
            simulated=True,
            device_fingerprint=device_fingerprint,
            region_graph_hash=f"transfer:{source}->{destination}:{nbytes}",
            shape=((nbytes,),),
            dtype=("uint8",),
            backend_implementation="mock_accel_copy_engine",
            kind=ProfileKind.TRANSFER,
            notes=("simulated virtual accelerator", f"source={source}", f"destination={destination}"),
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
        # Independent virtual streams: overlap cost is max(compute, transfer).
        cost = max(self.compute_delay_s, self.transfer_delay_s)
        timings = [cost for _ in range(max(1, samples))]
        return _summarize(
            timings,
            warm_up=warm_up,
            measured=False,
            simulated=True,
            device_fingerprint=device_fingerprint,
            region_graph_hash="overlap:mock_accel",
            backend_implementation="mock_accel",
            kind=ProfileKind.OVERLAP,
            notes=("simulated virtual accelerator",),
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
        timings = [1e-6 for _ in range(max(1, samples))]
        return _summarize(
            timings,
            warm_up=0,
            measured=False,
            simulated=True,
            device_fingerprint=device_fingerprint,
            region_graph_hash=f"memory:{nbytes}",
            backend_implementation="mock_accel",
            kind=ProfileKind.MEMORY,
            workspace_memory_bytes=nbytes,
            notes=("simulated virtual accelerator",),
        )
