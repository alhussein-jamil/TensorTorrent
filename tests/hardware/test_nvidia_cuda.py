"""Real NVIDIA CUDA coverage — skipped when no GPU is available."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

import tensortorrent as tt
from tensortorrent.cli.main import main
from tensortorrent.hardware.discovery import discover_resource_graph
from tensortorrent.validation.hardware import CheckStatus, validate_hardware

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.hardware,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="NVIDIA CUDA GPU required"),
]


def test_discovery_reports_nvidia_cuda_gpu() -> None:
    graph = discover_resource_graph()
    assert "cuda" in graph.backends_present
    gpus = graph.gpus()
    assert gpus, "expected at least one CUDA GPU in the resource graph"
    nvidia = [g for g in gpus if g.vendor == "nvidia" or g.backend_id == "cuda"]
    assert nvidia, f"expected NVIDIA/cuda compute nodes, got {[(g.id.name, g.vendor, g.backend_id) for g in gpus]}"
    assert any("cuda_gpu_" in g.id.name for g in nvidia)


def test_compile_places_on_cuda_and_matches_eager() -> None:
    # Size this so CUDA wins placement against CPU.
    model = nn.Sequential(nn.Linear(1024, 1024), nn.ReLU(), nn.Linear(1024, 8)).eval()
    x = torch.randn(32, 1024)
    with torch.no_grad():
        expected = model(x)
    compiled = tt.compile(model, (x,), config=tt.CompileConfig(allow_cpu=True, allow_gpu=True))
    try:
        devices = set(compiled.specialized.plan.devices_used)
        assert any(d.startswith("cuda_gpu_") for d in devices), f"expected CUDA placement, got {devices}"
        assert any("region_costs=measured" in n for n in compiled.specialized.plan.notes), (
            f"CUDA placements must be measured, notes={compiled.specialized.plan.notes}"
        )
        assert compiled.specialized.validation["regions_measured"] == compiled.specialized.validation["regions_total"]
        assert all(p.measured for p in compiled.specialized.plan.placements)
        torch.testing.assert_close(compiled(x), expected, atol=1e-4, rtol=1e-4, check_device=False)
        report = compiled.last_execution_report()
        assert report["region_count"] >= 1
        schedule = compiled.specialized.schedule
        assert schedule is not None
        for inst in schedule.instructions:
            if inst.attributes.get("simulated_until_validated") is True:
                assert "mock" in f"{inst.source}|{inst.destination}|{inst.resource}".lower()
    finally:
        compiled.close()


def test_forced_cuda_path_measures_and_runs() -> None:
    model = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 8)).eval()
    x = torch.randn(8, 256)
    with torch.no_grad():
        expected = model(x)
    compiled = tt.compile(
        model,
        (x,),
        config=tt.CompileConfig(allow_cpu=False, allow_gpu=True, use_torch_compile=False),
    )
    try:
        assert set(compiled.specialized.plan.devices_used) == {"cuda_gpu_0"}
        assert all(p.measured for p in compiled.specialized.plan.placements)
        out = compiled(x.cuda())
        assert out.device.type == "cuda", f"GPU plan must return CUDA tensors, got {out.device}"
        torch.testing.assert_close(out.cpu(), expected, atol=1e-4, rtol=1e-4)
        assert compiled.specialized.validation["cross_device_execution"] == "single_gpu"
    finally:
        compiled.close()


def test_doctor_marks_cuda_backend_available(tmp_path: Path) -> None:
    report = tmp_path / "doctor.json"
    assert main(["doctor", "--json", str(report)]) == 0
    payload = json.loads(report.read_text(encoding="utf-8"))
    by_name = {c["name"]: c for c in payload["checks"]}
    assert by_name["backend_available:cuda"]["status"] == "backend_available"
    numerics = by_name["numerical_equivalence_eager"]
    assert numerics["status"] == "numerical_correctness_validated"
    basic = next(c for c in payload["checks"] if c["name"].startswith("basic_execution:cuda_gpu_"))
    assert basic["status"] == "basic_execution_validated"
    assert "executed_matmul" in basic["detail"]


def test_autotune_on_cuda_places_on_nvidia_gpu(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    model = nn.Sequential(nn.Linear(1024, 1024), nn.ReLU(), nn.Linear(1024, 8)).eval()
    out = tmp_path / "artifact"
    compiled = tt.compile(model, (torch.randn(16, 1024),), artifact_dir=out)
    compiled.save(out)
    assert main(["autotune", "--force", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "cuda_gpu_" in printed
    assert "devices_used: cuda_gpu_" in printed or "cuda_gpu_0" in printed
    assert "region_costs=measured" in printed
    assert "priors_only" not in printed


def test_validate_hardware_executes_cuda_basic_path() -> None:
    report = validate_hardware(full=False, stress=False)
    cuda_exec = [c for c in report.checks if c.name.startswith("basic_execution:cuda_gpu_")]
    assert cuda_exec, "expected basic_execution check for cuda_gpu_*"
    assert all(c.status is CheckStatus.BASIC_EXECUTION_VALIDATED for c in cuda_exec)
    assert all("executed_matmul" in c.detail for c in cuda_exec)
    assert any(c.name == "backend_available:cuda" and c.status is CheckStatus.BACKEND_AVAILABLE for c in report.checks)


def test_cuda_collectives_select_nccl_when_available() -> None:
    import torch

    from tensortorrent.backends.communication import NcclComm, select_communication_backend

    caps = NcclComm().capabilities(("cuda_gpu_0", "cuda_gpu_1"))
    assert caps.available is True
    assert "allreduce" in caps.ops
    selected = select_communication_backend(("cuda_gpu_0", "cuda_gpu_1"))
    assert selected.backend_id == "nccl"
    out = selected.allreduce([torch.ones(4), torch.ones(4)], ("cuda_gpu_0", "cuda_gpu_1"))
    assert out.shape == (4,)
