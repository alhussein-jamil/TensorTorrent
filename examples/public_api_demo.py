"""Public API smoke documentation as executable examples."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

import streamcompiler as sc
from streamcompiler.hardware.discovery import discover_resource_graph
from streamcompiler.validation import validate_hardware


def demo() -> None:
    print("=== doctor excerpt ===")
    report = validate_hardware(full=False)
    print(report.render_text().splitlines()[0:8])

    print("\n=== resource graph ===")
    graph = discover_resource_graph()
    print(f"fingerprint={graph.fingerprint}")
    print(f"backends={graph.backends_present}")
    print(f"compute={list(graph.compute)}")

    print("\n=== compile + explain ===")
    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8))
    compiled = sc.compile(model, torch.randn(2, 16), artifact_dir=Path("artifacts/demo_api"))
    print(compiled.explain())


if __name__ == "__main__":
    demo()
