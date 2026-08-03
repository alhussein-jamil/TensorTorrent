#!/usr/bin/env python3
"""CPU-only native path microbench: compile once, many forwards, report stats.

Accelerator numbers are never claimed measured here — CPU only.
"""

from __future__ import annotations

import statistics
import time

import torch
import torch.nn as nn

import tensortorrent as tt
from tensortorrent.config import CompileConfig
from tensortorrent.native import require_native


def main() -> None:
    require_native()
    model = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 64)).eval()
    x = torch.randn(32, 256)
    compiled = tt.compile(
        model,
        (x,),
        config=CompileConfig(use_torch_compile=False, measure_regions=False),
    )
    try:
        # Warmup
        for _ in range(5):
            compiled(x)
        times: list[float] = []
        for _ in range(50):
            t0 = time.perf_counter()
            compiled(x)
            times.append(time.perf_counter() - t0)
        stats = compiled.last_report.parameter_store  # type: ignore[union-attr]
        print("native_runtime", stats.get("native_runtime"))
        print("native_data_plane", stats.get("native_data_plane"))
        print("native_artifact_reused", stats.get("native_artifact_reused"))
        print("median_forward_s", statistics.median(times))
        print("p95_forward_s", sorted(times)[int(0.95 * (len(times) - 1))])
        print("status", "measured" if stats.get("native_runtime") else "unknown")
    finally:
        compiled.close()


if __name__ == "__main__":
    main()
