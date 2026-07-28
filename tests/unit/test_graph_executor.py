"""Scheduling behaviour of the region executor."""

from __future__ import annotations

import dataclasses

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


def test_single_worker_uses_the_verified_static_order() -> None:
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
    assert executor.max_workers == 1
    assert executor._static_order is not None
    assert [r.region_id for r in executor._static_order] == list(branched.regions)


def test_out_of_order_regions_fall_back_to_the_dynamic_scheduler() -> None:
    """The static order is checked, not assumed, so a bad order still runs correctly."""
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
    assert executor._static_order is None, "reversed regions must not pass the topological check"

    flat_outputs, report = executor.run(shuffled.flatten_inputs((x,), {}))
    with torch.no_grad():
        torch.testing.assert_close(shuffled.unflatten_outputs(flat_outputs), model(x))
    assert len(report.events) == len(program.regions)


def test_resident_store_reports_no_prefetch_need() -> None:
    """Skipping prefetch bookkeeping must be driven by the store, not by a guess."""
    compiled = sc.compile(nn.Linear(8, 8).eval(), (torch.randn(2, 8),))
    assert compiled.executor.parameter_store.needs_prefetch is False
    assert compiled.executor._prefetch_enabled is False


def test_single_region_resident_models_use_the_fast_path() -> None:
    model = nn.Linear(8, 4).eval()
    x = torch.randn(2, 8)
    compiled = sc.compile(model, (x,))
    assert compiled.executor.uses_fast_path
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
    with torch.no_grad():
        torch.testing.assert_close(compiled(x), model(x))


def test_disabling_concurrency_fuses_branches_into_one_region() -> None:
    compiled = sc.compile(
        Branching().eval(),
        (torch.randn(2, 16),),
        config=sc.CompileConfig(allow_concurrent_regions=False),
    )
    assert len(compiled.regions) == 1
    assert compiled.executor.uses_fast_path
    assert compiled.program.metadata["force_single_region"] is True
