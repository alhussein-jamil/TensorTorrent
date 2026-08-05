"""Heterogeneous plans must not trust CPU-submodule concurrency probes."""

from __future__ import annotations

from types import SimpleNamespace

from tensortorrent.compile.pipeline import _decide_concurrency, _plan_is_cpu_accelerator, concurrency_budget
from tensortorrent.config import CompileConfig


def test_plan_is_cpu_accelerator_detects_mixed_devices() -> None:
    plan = SimpleNamespace(devices_used=("cpu_numa_0", "cuda_gpu_0"))
    assert _plan_is_cpu_accelerator(plan) is True
    assert _plan_is_cpu_accelerator(SimpleNamespace(devices_used=("cuda_gpu_0",))) is False


def test_hetero_concurrency_skips_cpu_submodule_measure(monkeypatch) -> None:
    from tensortorrent.compile import pipeline as pipe

    called: list[bool] = []

    def _boom(*_a, **_k):
        called.append(True)
        raise AssertionError("CPU submodule measure must not run for cpu+gpu plans")

    monkeypatch.setattr(pipe, "measure_concurrency_benefit", _boom)

    class _Prog:
        regions = (
            SimpleNamespace(region_id="region_0", depends_on=(), node_count=4),
            SimpleNamespace(region_id="region_1", depends_on=(), node_count=4),
            SimpleNamespace(region_id="region_2", depends_on=("region_0", "region_1"), node_count=1),
        )

    plan = SimpleNamespace(devices_used=("cpu_numa_0", "cuda_gpu_0"))
    machine = SimpleNamespace(
        compute={
            "cpu_numa_0": SimpleNamespace(backend_id="cpu", concurrency_limit=8),
            "cuda_gpu_0": SimpleNamespace(backend_id="cuda", concurrency_limit=1),
        }
    )
    decision = _decide_concurrency(
        _Prog(),
        {"region_0": (None,), "region_1": (None,), "region_2": (None,)},
        plan,
        machine,
        CompileConfig(),
    )
    assert called == []
    assert decision.enabled is True
    assert decision.workers >= 2
    assert decision.measured is False
    assert "heterogeneous" in decision.reason


def test_hetero_concurrency_budget_caps_at_device_classes() -> None:
    plan = SimpleNamespace(devices_used=("cpu_numa_0", "cuda_gpu_0"))
    machine = SimpleNamespace(
        compute={
            "cpu_numa_0": SimpleNamespace(backend_id="cpu", concurrency_limit=64),
            "cuda_gpu_0": SimpleNamespace(backend_id="cuda", concurrency_limit=1),
        }
    )
    assert concurrency_budget(plan, machine) == 2
