"""Operator microbenchmarks used during specialization."""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class OpBenchResult:
    op: str
    device: str
    dtype: str
    shape: tuple[int, ...]
    latency_s: float
    measured: bool
    notes: str = ""


def benchmark_matmul(device: str, dtype: str, n: int = 512) -> OpBenchResult:
    import torch

    torch_device = torch.device("cpu")
    if device.startswith("cuda_"):
        if not torch.cuda.is_available():
            return OpBenchResult("mm", device, dtype, (n, n), float("inf"), False, "cuda unavailable")
        torch_device = torch.device(f"cuda:{device.rsplit('_', 1)[-1]}")
    dt = getattr(torch, dtype, torch.float32)
    a = torch.randn(n, n, device=torch_device, dtype=dt)
    b = torch.randn(n, n, device=torch_device, dtype=dt)
    if torch_device.type == "cuda":
        torch.cuda.synchronize()
    for _ in range(3):
        torch.mm(a, b)
    if torch_device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    iters = 10
    for _ in range(iters):
        torch.mm(a, b)
    if torch_device.type == "cuda":
        torch.cuda.synchronize()
    return OpBenchResult("mm", device, dtype, (n, n), (time.perf_counter() - start) / iters, True)


def representative_shape_buckets() -> tuple[tuple[int, ...], ...]:
    return (
        (1, 128),
        (1, 512),
        (1, 4096),
        (2, 1024),
        (4, 1024),
        (8, 2048),
    )
