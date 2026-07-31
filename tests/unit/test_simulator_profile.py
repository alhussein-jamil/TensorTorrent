"""Specialization profile carries simulator and validation metadata."""

from __future__ import annotations

import torch
import torch.nn as nn

import streamcompiler as sc


def test_specialization_profile_marks_simulator_analytic() -> None:
    model = nn.Linear(8, 4).eval()
    x = torch.randn(2, 8)
    compiled = sc.compile(model, (x,))
    try:
        sim = compiled.specialized.profile["simulator"]
        assert sim["simulated"] is True
        assert "makespan_s" in sim
        assert "eviction_pressure_events" in sim
        assert "transfer_landed_events" in sim
        assert "peak_bytes" in sim
        assert isinstance(sim["peak_bytes"], dict)
        assert compiled.specialized.validation["cross_device_execution"] in {
            "multi_gpu",
            "cpu_gpu",
            "single_gpu",
            "cpu_only",
        }
        assert "simulated_makespan_s" in compiled.specialized.validation
        assert "residency" in compiled.specialized.profile
    finally:
        compiled.close()
