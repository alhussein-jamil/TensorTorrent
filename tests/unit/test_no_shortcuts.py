"""Reject architectural shortcuts that homogenize heterogeneous machines."""

from __future__ import annotations

from streamcompiler.backends import all_backends
from streamcompiler.ir.resource_graph import ensure_host_staged_fallbacks
from streamcompiler.planner import enumerate_plan_strategies


def test_multiple_accelerator_backends_registered() -> None:
    ids = {b.backend_id for b in all_backends()}
    for required in ("cpu", "cuda", "rocm", "mps", "sycl", "opencl", "vulkan"):
        assert required in ids


def test_planner_strategy_catalog_not_cuda_only() -> None:
    strategies = enumerate_plan_strategies()
    assert "cpu_only" in strategies
    assert "tensor_partition_gpus_and_cpus" in strategies
    assert "shared_weight_streaming" in strategies


def test_host_staged_helper_exists_for_missing_p2p() -> None:
    # Imported symbol must remain part of the public IR API.
    assert callable(ensure_host_staged_fallbacks)


def test_no_backend_returns_a_fake_success_dictionary() -> None:
    """Production sources must not contain status-dictionary stand-ins."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[2] / "src" / "streamcompiler"
    offenders: list[str] = []
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for marker in ('"status": "ok"', '"status": "planned"', "'status': 'ok'"):
            if marker in text:
                offenders.append(f"{path.relative_to(root)}: {marker}")
    assert not offenders, f"fake success payloads found: {offenders}"


def test_unavailable_backends_raise_instead_of_reporting_success() -> None:
    import pytest
    import torch

    from streamcompiler.backends.base import CompiledRegion, KernelCandidate, RegionSource
    from streamcompiler.errors import StreamCompilerError

    for backend in all_backends():
        if backend.backend_id == "cpu" or backend.available():
            continue
        source = RegionSource(region_id="probe", module=torch.nn.Identity())
        candidate = KernelCandidate("probe", f"{backend.backend_id}_0", backend.backend_id, "k", "float32")
        with pytest.raises(StreamCompilerError) as compile_error:
            backend.compile(source, candidate)
        assert "not available" in str(compile_error.value) or "not implemented" in str(compile_error.value)
        region = CompiledRegion(
            region_id="probe",
            device=f"{backend.backend_id}_0",
            backend_id=backend.backend_id,
            executable=torch.nn.Identity(),
            dtype="float32",
        )
        with pytest.raises(StreamCompilerError) as execute_error:
            backend.execute(region, (torch.randn(2),))
        assert "not available" in str(execute_error.value) or "not implemented" in str(execute_error.value)
