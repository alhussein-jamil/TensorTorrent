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
