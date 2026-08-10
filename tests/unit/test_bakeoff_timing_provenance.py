"""Bakeoff timing must label predicted streaming fallback as not measured."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from tensortorrent.compile.bakeoff import BakeoffTiming, safe_time_executor
from tensortorrent.config import CompileConfig
from tensortorrent.ir.graph import OpCode
from tensortorrent.ir.resource_graph import ResourceGraph


def _streaming_schedule() -> Any:
    return SimpleNamespace(
        instructions=(
            SimpleNamespace(
                opcode=OpCode.LOAD,
                attributes={"kind": "parameter_materialize"},
            ),
        )
    )


def test_safe_time_executor_predicted_fallback_is_not_measured() -> None:
    predicted = 0.042
    specialized = SimpleNamespace(
        schedule=_streaming_schedule(),
        plan=SimpleNamespace(predicted_latency_s=predicted, prefetch_distance=1),
        bindings={},
    )
    timing = safe_time_executor(
        program=SimpleNamespace(),  # unused on predicted path
        specialized=specialized,  # type: ignore[arg-type]
        flat_inputs=[],
        config=CompileConfig(),
        machine=ResourceGraph(fingerprint="test"),
        workers=1,
        intraop_threads=0,
    )
    assert isinstance(timing, BakeoffTiming)
    assert timing.measured is False
    assert timing.seconds == predicted


def test_safe_time_executor_real_measure_is_marked_measured(monkeypatch: Any) -> None:
    from tensortorrent.compile import bakeoff as bakeoff_mod

    specialized = SimpleNamespace(
        schedule=SimpleNamespace(instructions=()),
        plan=SimpleNamespace(predicted_latency_s=9.9, prefetch_distance=1),
        bindings={},
    )

    def _fake_time_executor(*_a: Any, **_k: Any) -> float:
        return 0.0125

    monkeypatch.setattr(bakeoff_mod, "time_executor", _fake_time_executor)
    timing = safe_time_executor(
        program=SimpleNamespace(),  # type: ignore[arg-type]
        specialized=specialized,  # type: ignore[arg-type]
        flat_inputs=[1],
        config=CompileConfig(),
        machine=ResourceGraph(fingerprint="test"),
        workers=1,
        intraop_threads=0,
    )
    assert timing.measured is True
    assert timing.seconds == 0.0125
