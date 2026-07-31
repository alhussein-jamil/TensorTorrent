"""Reject architectural shortcuts that homogenize heterogeneous machines."""

from __future__ import annotations

from streamcompiler.backends import all_backends
from streamcompiler.ir.resource_graph import ensure_host_staged_fallbacks
from streamcompiler.planner import enumerate_plan_strategies


def test_multiple_accelerator_backends_registered() -> None:
    ids = {b.backend_id for b in all_backends()}
    # Supported discovery surface. Unsupported accelerator stubs removed.
    for required in ("cpu", "cuda", "rocm", "mock_accel"):
        assert required in ids
    for removed in ("mps", "sycl", "opencl", "vulkan"):
        assert removed not in ids


def test_planner_strategy_catalog_covers_cpu_and_gpu() -> None:
    strategies = enumerate_plan_strategies()
    assert "cpu_only" in strategies
    assert "single_gpu" in strategies
    assert "multi_gpu" in strategies
    assert "tensor_partition_gpus_and_cpus" in strategies
    assert "shared_weight_streaming" in strategies


def test_host_staged_helper_exists_for_missing_p2p() -> None:
    assert callable(ensure_host_staged_fallbacks)


def test_no_backend_returns_a_fake_success_dictionary() -> None:
    """Production sources must not contain status-dictionary stand-ins."""
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parents[2] / "python" / "streamcompiler"
    offenders: list[str] = []
    pattern = re.compile(r"""["']status["']\s*:\s*["'](?:ok|planned\w*)["']""")
    for path in root.rglob("*.py"):
        # HTTP health payloads in serve/ are product API, not backend stand-ins.
        if "serve" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        for match in pattern.finditer(text):
            offenders.append(f"{path.relative_to(root)}: {match.group(0)}")
    assert not offenders, f"fake success payloads found: {offenders}"


def test_host_staged_allreduce_sums_cpu_tensors() -> None:
    import torch

    from streamcompiler.backends.communication import HostStagedComm

    a = torch.ones(4)
    b = torch.full((4,), 2.0)
    out = HostStagedComm().allreduce([a, b], ("cpu_numa_0", "cpu_numa_1"))
    assert torch.equal(out, torch.full((4,), 3.0))
    assert out.dtype == torch.float32


def test_host_staged_allreduce_preserves_integer_dtype() -> None:
    import torch

    from streamcompiler.backends.communication import HostStagedComm

    a = torch.ones(3, dtype=torch.int64)
    b = torch.full((3,), 4, dtype=torch.int64)
    out = HostStagedComm().allreduce([a, b], ("cpu_numa_0",))
    assert out.dtype == torch.int64
    assert torch.equal(out, torch.full((3,), 5, dtype=torch.int64))


def test_unavailable_backends_raise_instead_of_reporting_success() -> None:
    import pytest
    import torch

    from streamcompiler.backends.base import KernelCandidate, RegionSource
    from streamcompiler.errors import StreamCompilerError

    for backend in all_backends():
        if backend.backend_id in {"cpu", "mock_accel"} or backend.available():
            continue
        source = RegionSource(region_id="probe", module=torch.nn.Identity())
        candidate = KernelCandidate("probe", f"{backend.backend_id}_0", backend.backend_id, "k", "float32")
        with pytest.raises(StreamCompilerError):
            backend.compile(source, candidate)
