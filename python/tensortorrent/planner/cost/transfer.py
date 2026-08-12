"""Size- and contention-dependent transfer cost model.

Theoretical peak bandwidth is never used as the sole predictor. Prefer measured
samples keyed by (source, destination, bytes, concurrency).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class TransferSample:
    source: str
    destination: str
    nbytes: int
    concurrency: int
    latency_s: float
    measured: bool
    notes: str = ""


@dataclass
class TransferModel:
    """Piecewise model: time ≈ alpha + bytes / beta, with contention multiplier."""

    source: str
    destination: str
    alpha_s: float = 0.0
    beta_bytes_per_s: float | None = None
    contention_factor: float = 1.0
    samples: list[TransferSample] = field(default_factory=list)
    measured: bool = False

    def predict(self, nbytes: int, concurrency: int = 1) -> float:
        if self.beta_bytes_per_s and self.beta_bytes_per_s > 0:
            base = self.alpha_s + nbytes / self.beta_bytes_per_s
        elif self.samples:
            # Nearest-neighbor in log-bytes space.
            best = min(self.samples, key=lambda s: abs(np.log2(max(1, s.nbytes)) - np.log2(max(1, nbytes))))
            scale = nbytes / max(1, best.nbytes)
            base = best.latency_s * scale
        else:
            # Explicit unknown prior — not a peak-bandwidth claim and not measured.
            # ~4 GB/s host-ish copy plus 100 µs setup; callers must treat as prior.
            base = 1e-4 + nbytes / (4e9)
            self.measured = False
        return base * max(1.0, float(concurrency)) * self.contention_factor


def measure_host_copy(
    source: str,
    destination: str,
    sizes: tuple[int, ...] = (1 << 20, 8 << 20, 64 << 20),
    concurrency: int = 1,
) -> TransferModel:
    """Measure host memory copies as a baseline transfer model."""
    samples: list[TransferSample] = []
    for nbytes in sizes:
        src = np.empty(nbytes, dtype=np.uint8)
        dst = np.empty(nbytes, dtype=np.uint8)
        src.fill(1)
        # Warmup
        np.copyto(dst, src)
        start = time.perf_counter()
        iters = 5
        for _ in range(iters):
            np.copyto(dst, src)
        elapsed = (time.perf_counter() - start) / iters
        samples.append(
            TransferSample(
                source=source,
                destination=destination,
                nbytes=nbytes,
                concurrency=concurrency,
                latency_s=elapsed,
                measured=True,
                notes="numpy host copy",
            )
        )
    # Fit alpha/beta with two largest points when possible.
    model = TransferModel(source=source, destination=destination, samples=samples, measured=True)
    if len(samples) >= 2:
        a, b = samples[-2], samples[-1]
        denom = b.nbytes - a.nbytes
        if denom > 0 and b.latency_s > a.latency_s:
            beta = denom / (b.latency_s - a.latency_s)
            alpha = a.latency_s - a.nbytes / beta
            model.alpha_s = max(0.0, alpha)
            model.beta_bytes_per_s = beta
    return model


def transfer_time(
    model: TransferModel | None,
    source: str,
    destination: str,
    nbytes: int,
    concurrency: int = 1,
    topology: str | None = None,
) -> float:
    """Public cost entrypoint used by the planner/simulator."""
    if model is None:
        # Unknown path prior; topology string is recorded by callers for debugging.
        _ = topology
        return TransferModel(source, destination).predict(nbytes, concurrency)
    return model.predict(nbytes, concurrency)


# Analytic priors until profiling installs a measured compute multiplier.
# Incremental slowdown per extra contender.
COMPUTE_SLOWDOWN_PER_EXTRA_TASK = 0.05
TRANSFER_SLOWDOWN_PER_EXTRA_TRANSFER = 0.15
TRANSFER_SLOWDOWN_PER_COMPUTE_TASK = 0.05
STORAGE_SLOWDOWN_PER_EXTRA_TASK = 0.20
STORAGE_SLOWDOWN_PER_TRANSFER = 0.05


@dataclass
class ContentionFactors:
    compute: float = 1.0
    transfer: float = 1.0
    storage: float = 1.0


_MEASURED_COMPUTE: float | None = None


def set_measured_compute_contention(factor: float | None) -> None:
    """Process-local measured compute contention multiplier."""
    global _MEASURED_COMPUTE
    _MEASURED_COMPUTE = None if factor is None else max(1.0, float(factor))


def concurrent_slowdown(
    *,
    active_compute: int,
    active_transfers: int,
    active_storage: int,
) -> ContentionFactors:
    """Return multiplicative slowdowns under concurrent pressure."""
    base_compute = 1.0 + COMPUTE_SLOWDOWN_PER_EXTRA_TASK * max(0, active_compute - 1)
    if _MEASURED_COMPUTE is not None and active_compute > 1:
        base_compute = max(base_compute, _MEASURED_COMPUTE)
    transfer = (
        1.0
        + TRANSFER_SLOWDOWN_PER_EXTRA_TRANSFER * max(0, active_transfers - 1)
        + TRANSFER_SLOWDOWN_PER_COMPUTE_TASK * active_compute
    )
    storage = (
        1.0
        + STORAGE_SLOWDOWN_PER_EXTRA_TASK * max(0, active_storage - 1)
        + STORAGE_SLOWDOWN_PER_TRANSFER * active_transfers
    )
    return ContentionFactors(compute=base_compute, transfer=transfer, storage=storage)
