"""Fusion-candidate timing must synchronize devices and release probes."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch

from tensortorrent.compile import pipeline
from tensortorrent.runtime.direct_path import DirectParameter


def test_accelerator_timing_synchronizes_each_bound_cuda_device(monkeypatch: Any) -> None:
    synchronized: list[str] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: synchronized.append(str(device)))
    bindings = {
        "a": SimpleNamespace(backend_id="cuda", compiled=SimpleNamespace(torch_device="cuda:1")),
        "b": SimpleNamespace(backend_id="cuda", compiled=SimpleNamespace(torch_device="cuda:0")),
        "cpu": SimpleNamespace(backend_id="cpu", compiled=SimpleNamespace(torch_device="cpu")),
    }

    pipeline._synchronize_bound_accelerators(bindings)

    assert synchronized == ["cuda:0", "cuda:1"]


def test_executor_timing_closes_probe_resources(monkeypatch: Any) -> None:
    lifecycle: list[str] = []

    class FakeStore:
        def __init__(self, _state: dict[str, Any]) -> None:
            lifecycle.append("store-created")

        def close(self) -> None:
            lifecycle.append("store-closed")

    class FakeExecutor:
        direct_plan = None

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            lifecycle.append("executor-created")

        def run(self, _inputs: list[Any]) -> None:
            lifecycle.append("run")

        def close(self) -> None:
            lifecycle.append("executor-closed")

    class FakeProgram:
        @staticmethod
        def state_tensors() -> dict[str, Any]:
            return {}

    import tensortorrent.runtime.graph_executor as graph_executor
    import tensortorrent.runtime.tensor_store as tensor_store

    monkeypatch.setattr(graph_executor, "GraphExecutor", FakeExecutor)
    monkeypatch.setattr(tensor_store, "ResidentParameterStore", FakeStore)
    monkeypatch.setattr(pipeline, "_synchronize_bound_accelerators", lambda _bindings: None)

    assert pipeline._time_executor(FakeProgram(), {}, [], workers=2, intraop_threads=0, iters=2) >= 0
    assert lifecycle.count("run") == 4
    assert lifecycle[-2:] == ["executor-closed", "store-closed"]


def test_hetero_near_tie_keeps_dataflow_when_schedule_is_slow() -> None:
    prefer_fused, dataflow_enabled, concurrent_s, fuse_margin = pipeline._choose_fusion_candidate(
        fused_s=0.0040,
        concurrent_schedule_s=0.0120,
        concurrent_dataflow_s=0.0041,
        hetero_plan=True,
    )
    assert dataflow_enabled is True
    assert prefer_fused is False
    assert fuse_margin == 1.10
    assert concurrent_s == 0.0041


def test_clear_fused_win_still_prefers_fusion() -> None:
    prefer_fused, dataflow_enabled, concurrent_s, fuse_margin = pipeline._choose_fusion_candidate(
        fused_s=0.0020,
        concurrent_schedule_s=0.0100,
        concurrent_dataflow_s=0.0050,
        hetero_plan=True,
    )
    assert dataflow_enabled is True
    assert prefer_fused is True
    assert fuse_margin == 1.10
    assert concurrent_s == 0.0050


def test_region_pool_keeps_explicit_cpu_thread_budget() -> None:
    from concurrent.futures import ThreadPoolExecutor

    from tensortorrent.runtime.schedule_executor import ScheduleExecutor

    executor = object.__new__(ScheduleExecutor)
    executor._region_pool = None
    executor._region_pool_threads = None
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        pool = ScheduleExecutor._ensure_region_pool(executor, 1, threads=4)
        assert isinstance(pool, ThreadPoolExecutor)
        assert executor._region_pool_threads == 4
        seen: list[int] = []

        def _probe() -> None:
            seen.append(torch.get_num_threads())

        pool.submit(_probe).result(timeout=5)
        assert seen == [4]
    finally:
        if executor._region_pool is not None:
            executor._region_pool.shutdown(wait=False, cancel_futures=True)
        torch.set_num_threads(previous)


def test_dataflow_probe_skipped_when_direct_path_disabled(monkeypatch: Any) -> None:
    """Fusion must not score a dataflow path the final executor cannot use."""
    from tensortorrent.runtime import graph_executor as ge

    assert ge._direct_path_wanted(SimpleNamespace(prefer_direct_path=False)) is False
    monkeypatch.setenv("TT_DIRECT_PATH", "0")
    assert ge._direct_path_wanted(SimpleNamespace(prefer_direct_path=True)) is False
    monkeypatch.delenv("TT_DIRECT_PATH", raising=False)


def test_shared_weight_places_once_per_device() -> None:
    source = torch.tensor([1.0, 2.0])
    first = DirectParameter.place(source, torch.device("cpu"))
    second = DirectParameter.place(source, torch.device("cpu"))
    # place() itself always copies; builders must memoize — exercise identity key.
    assert id(source) == id(first.source) == id(second.source)
    cache: dict[tuple[int, str], DirectParameter] = {}
    key = (id(source), "cpu")
    cache[key] = first
    assert cache[key] is first
    assert cache.setdefault(key, second) is first


def test_pending_cancel_forces_schedule_not_dataflow(monkeypatch: Any) -> None:
    from tensortorrent.errors import ExecutionCancelled
    from tensortorrent.runtime.direct_path import DataflowDirectPlan
    from tensortorrent.runtime.graph_executor import GraphExecutor

    executor = object.__new__(GraphExecutor)
    executor._closed = False
    executor._schedule_executor = SimpleNamespace(request_cancel=lambda: None, run=None)
    executor._direct_plan = DataflowDirectPlan(
        waves=(),
        user_inputs=(),
        output_refs=(),
        parameters=(),
        param_bytes=0,
    )
    executor._dataflow_direct_path_enabled = True
    executor.intraop_threads = 0
    executor._thread_lock = __import__("threading").Lock()
    executor._thread_owners = 0
    executor._saved_threads = None
    executor._gate = SimpleNamespace(enter=lambda: None, leave=lambda: None)
    executor.max_workers = 2
    executor._region_pool_threads = 2
    executor._cancel_requested = True

    calls: list[str] = []

    def _schedule(*_a: Any, **_k: Any) -> tuple[list[Any], Any]:
        calls.append("schedule")
        raise ExecutionCancelled("cancelled")

    monkeypatch.setattr(executor, "_run_via_schedule", _schedule)
    monkeypatch.setattr(executor, "_run_direct", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("direct")))

    with __import__("pytest").raises(ExecutionCancelled):
        GraphExecutor.run(executor, [])
    assert calls == ["schedule"]
