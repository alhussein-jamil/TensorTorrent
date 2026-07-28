"""Benchmark harness scaffolding (results are measured, never fabricated)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn as nn


def run_eager_baseline(model: nn.Module, x: torch.Tensor, warmup: int = 5, iters: int = 20) -> dict:
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        start = time.perf_counter()
        for _ in range(iters):
            model(x)
        latency = (time.perf_counter() - start) / iters
    return {"latency_s": latency, "framework": "eager_pytorch", "measured": True}


def main() -> None:
    out = Path("artifacts/benchmarks")
    out.mkdir(parents=True, exist_ok=True)
    model = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 256))
    x = torch.randn(8, 256)
    result = run_eager_baseline(model, x)
    (out / "eager_tiny.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
