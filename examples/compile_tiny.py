"""Compile a tiny model and print the specialized plan explanation."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

import tensortorrent as tt


def main() -> None:
    model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 10))
    model.eval()
    x = torch.randn(4, 32)
    compiled = tt.compile(
        model,
        x,
        config=tt.CompileConfig(objective=tt.Objective.LATENCY),
        artifact_dir=Path("artifacts/example_tiny"),
    )
    print(compiled.explain())
    compiled.visualize("artifacts/example_tiny/plan.html")
    print("wrote artifacts/example_tiny/")


if __name__ == "__main__":
    main()
