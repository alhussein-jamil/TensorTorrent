"""End-to-end compile → specialize → explain → execute on the local machine."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

import streamcompiler as sc


def test_e2e_cpu_compile_and_execute(tmp_path: Path) -> None:
    model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 8))
    compiled = sc.compile(
        model,
        torch.randn(2, 32),
        config=sc.CompileConfig(objective=sc.Objective.LATENCY, profile_level="coarse"),
        artifact_dir=tmp_path / "art",
    )
    text = compiled.explain()
    assert "devices_used" in text or "cpu_numa" in text
    compiled.visualize(str(tmp_path / "plan.html"))
    assert (tmp_path / "plan.html").exists()
    assert (tmp_path / "plan.trace.json").exists()
    result = compiled()
    assert result["results"]
    assert (tmp_path / "art" / "portable.json").exists()
