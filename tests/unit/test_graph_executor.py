"""Scheduling behaviour of the region executor."""

from __future__ import annotations

import dataclasses

import pytest
import torch
import torch.nn as nn

import streamcompiler as sc
from streamcompiler.runtime.graph_executor import GraphExecutor
from streamcompiler.runtime.tensor_store import ResidentParameterStore


class Branching(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Linear(16, 16)
        self.left = nn.Linear(16, 16)
        self.right = nn.Linear(16, 16)
        self.head = nn.Linear(16, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.stem(x))
        return self.head(torch.relu(self.left(h)) + torch.tanh(self.right(h)))


def test_single_worker_builds_schedule_from_bindings() -> None:
    branched = sc.compile(
        Branching().eval(),
        (torch.randn(2, 16),),
        config=sc.CompileConfig(max_concurrent_regions=2),
    )
    assert len(branched.regions) > 1
    executor = GraphExecutor(
        branched.program,
        branched.executor.bindings,
        parameter_store=ResidentParameterStore(branched.program.state_tensors()),
        max_workers=1,
    )
    assert executor.uses_schedule_path
    assert executor.schedule is not None
    assert executor.max_workers == 1


def test_out_of_order_regions_still_run_via_schedule_deps() -> None:
    """Compute order may differ from source region order; deps alone serialize."""
    model = Branching().eval()
    x = torch.randn(2, 16)
    compiled = sc.compile(model, (x,), config=sc.CompileConfig(max_concurrent_regions=2))
    program = compiled.program
    shuffled = dataclasses.replace(program, regions=tuple(reversed(program.regions)))

    executor = GraphExecutor(
        shuffled,
        compiled.executor.bindings,
        parameter_store=ResidentParameterStore(shuffled.state_tensors()),
    )
    assert executor.uses_schedule_path

    flat_outputs, report = executor.run(shuffled.flatten_inputs((x,), {}))
    with torch.no_grad():
        torch.testing.assert_close(shuffled.unflatten_outputs(flat_outputs), model(x))
    assert len(report.events) == len(program.regions)


def test_resident_store_reports_no_prefetch_need() -> None:
    """Skipping prefetch bookkeeping must be driven by the store, not by a guess."""
    compiled = sc.compile(nn.Linear(8, 8).eval(), (torch.randn(2, 8),))
    assert compiled.executor.parameter_store.needs_prefetch is False
    assert compiled.executor._prefetch_enabled is False


def test_single_region_resident_models_use_the_schedule_path() -> None:
    model = nn.Linear(8, 4).eval()
    x = torch.randn(2, 8)
    compiled = sc.compile(model, (x,))
    assert compiled.executor.uses_schedule_path
    assert compiled.executor.schedule is not None
    with torch.no_grad():
        expected = model(x)
    torch.testing.assert_close(compiled(x), expected)
    torch.testing.assert_close(compiled(x), expected)


def test_streaming_store_disables_the_fast_path() -> None:
    model = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 8)).eval()
    x = torch.randn(4, 64)
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    # Half the model, but large enough for the biggest single region after splitting.
    compiled = sc.compile(
        model,
        (x,),
        config=sc.CompileConfig(
            ram_budget_bytes=max(total // 2, 18_000),
            max_region_nodes=2,
            prefetch_distance=1,
        ),
    )
    assert compiled.executor.parameter_store.needs_prefetch is True
    assert not compiled.executor.uses_fast_path
    schedule = compiled.specialized.schedule
    assert schedule is not None
    by_name = {i.name: i for i in schedule.instructions}
    # Prefetch of region_1 must not race ahead of region_0 Load (budget steal).
    prefetch_1 = by_name.get("prefetch::region_1")
    if prefetch_1 is not None:
        assert "load::region_0" in prefetch_1.depends_on
    with torch.no_grad():
        for _ in range(5):
            torch.testing.assert_close(compiled(x), model(x))
    compiled.close()


def test_disabling_concurrency_fuses_branches_into_one_region() -> None:
    compiled = sc.compile(
        Branching().eval(),
        (torch.randn(2, 16),),
        config=sc.CompileConfig(allow_concurrent_regions=False),
    )
    assert len(compiled.regions) == 1
    assert compiled.executor.uses_schedule_path
    assert compiled.program.metadata["force_single_region"] is True


def test_schedule_path_records_real_region_durations() -> None:
    """Multi-region plans must time each Compute, not stamp identical clocks."""
    model = Branching().eval()
    x = torch.randn(4, 16)
    compiled = sc.compile(model, (x,), config=sc.CompileConfig(max_concurrent_regions=2))
    assert len(compiled.regions) > 1
    executor = GraphExecutor(
        compiled.program,
        compiled.executor.bindings,
        parameter_store=ResidentParameterStore(compiled.program.state_tensors()),
        max_workers=1,
    )
    assert executor.uses_schedule_path
    _, report = executor.run(compiled.program.flatten_inputs((x,), {}))
    assert len(report.events) == len(compiled.regions)
    assert all(event.duration_s >= 0.0 for event in report.events)
    assert sum(event.duration_s for event in report.events) > 0.0
    assert report.wall_time_s >= sum(event.duration_s for event in report.events) - 1e-3


class _Boom(RuntimeError):
    pass


def _raiser(*_args: object, **_kwargs: object) -> object:
    raise _Boom("region exploded")


def test_exception_in_a_region_propagates_out_of_the_call() -> None:
    """A raising region must fail the call loudly, on both the single- and
    multi-worker dispatch paths, not be swallowed or return a partial result."""
    model = Branching().eval()
    x = torch.randn(2, 16)
    compiled = sc.compile(model, (x,), config=sc.CompileConfig(max_concurrent_regions=2))
    try:
        region_id = compiled.program.regions[-1].region_id
        bindings = dict(compiled.executor.bindings)
        original = bindings[region_id]
        broken_compiled = dataclasses.replace(original.compiled, executable=_raiser)
        bindings[region_id] = dataclasses.replace(original, compiled=broken_compiled)
        for max_workers in (1, 2):
            executor = GraphExecutor(
                compiled.program,
                bindings,
                parameter_store=ResidentParameterStore(compiled.program.state_tensors()),
                max_workers=max_workers,
            )
            with pytest.raises(_Boom, match="region exploded"):
                executor.run(compiled.program.flatten_inputs((x,), {}))
    finally:
        compiled.close()


def test_repeated_calls_do_not_grow_tensor_directory_state() -> None:
    """Repeated forward calls must not leak per-call tensor records forever."""
    model = Branching().eval()
    x = torch.randn(2, 16)
    compiled = sc.compile(model, (x,), config=sc.CompileConfig(max_concurrent_regions=2))
    try:
        with torch.no_grad():
            for _ in range(5):
                compiled(x)
        directory = compiled.executor.tensor_directory
        first_size = len(directory.snapshot())
        with torch.no_grad():
            for _ in range(20):
                compiled(x)
        second_size = len(directory.snapshot())
        assert second_size <= first_size, "tensor directory must not accumulate one record per call forever"
    finally:
        compiled.close()


def test_request_cancel_aborts_before_next_region() -> None:
    """Cancel mid multi-region run raises ExecutionCancelled and leaves executor reusable."""
    from streamcompiler.errors import ExecutionCancelled

    model = Branching().eval()
    x = torch.randn(2, 16)
    compiled = sc.compile(model, (x,), config=sc.CompileConfig(max_concurrent_regions=2))
    try:
        assert len(compiled.regions) > 1
        executor = compiled.executor
        assert executor.uses_schedule_path
        assert executor._schedule_executor is not None
        seen: list[str] = []
        originals = dict(executor._schedule_executor._callables)

        def _wrap(region_id: str, call: object):
            def wrapped(*args: object, **kwargs: object) -> object:
                seen.append(region_id)
                if len(seen) == 1:
                    executor.request_cancel()
                return call(*args, **kwargs)

            return wrapped

        executor._schedule_executor._callables.clear()
        executor._schedule_executor._callables.update({rid: _wrap(rid, call) for rid, call in originals.items()})

        with pytest.raises(ExecutionCancelled, match="cancelled"):
            executor.run(compiled.program.flatten_inputs((x,), {}))
        assert len(seen) >= 1
        assert len(seen) < len(compiled.regions)

        # Next call must work after a cancel.
        executor._schedule_executor._callables.clear()
        executor._schedule_executor._callables.update(originals)
        outs, _report = executor.run(compiled.program.flatten_inputs((x,), {}))
        assert len(outs) == 1
        torch.testing.assert_close(outs[0], model(x), atol=1e-5, rtol=1e-5)
    finally:
        compiled.close()


def test_request_cancel_before_schedule_run() -> None:
    from streamcompiler.errors import ExecutionCancelled

    model = nn.Linear(8, 4).eval()
    x = torch.randn(2, 8)
    compiled = sc.compile(model, (x,))
    try:
        assert compiled.executor.uses_schedule_path
        compiled.executor.request_cancel()
        with pytest.raises(ExecutionCancelled, match="cancelled"):
            compiled.executor.run(compiled.program.flatten_inputs((x,), {}))
        # Flag cleared by the abort; subsequent calls succeed.
        out, _ = compiled.executor.run(compiled.program.flatten_inputs((x,), {}))
        torch.testing.assert_close(out[0], model(x), atol=1e-5, rtol=1e-5)
    finally:
        compiled.close()


def test_compiled_module_request_cancel_is_public() -> None:
    model = nn.Linear(8, 4).eval()
    x = torch.randn(2, 8)
    compiled = sc.compile(model, (x,))
    try:
        compiled.request_cancel()
        with pytest.raises(sc.ExecutionCancelled):
            compiled(x)
        torch.testing.assert_close(compiled(x), model(x), atol=1e-5, rtol=1e-5)
    finally:
        compiled.close()
