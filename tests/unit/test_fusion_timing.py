"""Fusion-candidate timing must synchronize devices and release probes."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch

from tensortorrent.compile import pipeline


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
