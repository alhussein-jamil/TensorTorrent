"""Size- and contention-dependent transfer cost model.

Theoretical peak bandwidth is never used as the sole predictor. Prefer measured
samples keyed by (source, destination, bytes, concurrency).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

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
            # Explicit unknown prior — not a peak-bandwidth claim.
            base = 1e-4 + nbytes / (4e9)
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


def measure_torch_device_transfer(
    src_device: str,
    dst_device: str,
    sizes: tuple[int, ...] = (1 << 20, 8 << 20),
) -> TransferModel | None:
    """Measure torch tensor transfers when devices exist; otherwise return None."""
    try:
        import torch
    except ImportError:
        return None

    def resolve(name: str) -> Any:
        if name.startswith("cuda_gpu_") or name.startswith("cuda_vram_"):
            idx = int(name.rsplit("_", 1)[-1])
            if not torch.cuda.is_available():
                return None
            return torch.device(f"cuda:{idx}")
        if name.startswith("cpu") or name.startswith("numa") or name.startswith("pinned"):
            return torch.device("cpu")
        return None

    src = resolve(src_device)
    dst = resolve(dst_device)
    if src is None or dst is None:
        return None
    if (src.type == "cuda" or dst.type == "cuda") and not torch.cuda.is_available():
        return None

    samples: list[TransferSample] = []
    for nbytes in sizes:
        nelem = max(1, nbytes // 4)
        x = torch.empty(nelem, device=src, dtype=torch.float32)
        if src.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        iters = 5
        for _ in range(iters):
            y = x.to(dst, non_blocking=False)
            if dst.type == "cuda":
                torch.cuda.synchronize()
            del y
        elapsed = (time.perf_counter() - start) / iters
        samples.append(
            TransferSample(
                source=src_device,
                destination=dst_device,
                nbytes=nbytes,
                concurrency=1,
                latency_s=elapsed,
                measured=True,
                notes="torch.Tensor.to",
            )
        )
    model = TransferModel(source=src_device, destination=dst_device, samples=samples, measured=True)
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
