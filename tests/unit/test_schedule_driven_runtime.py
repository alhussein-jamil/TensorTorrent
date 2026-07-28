"""Schedule-driven releases, physical reuse, and activation spill."""

from __future__ import annotations

import torch
import torch.nn as nn

import streamcompiler as sc
from streamcompiler.ir.graph import OpCode
from streamcompiler.runtime.activation_spill import reload_spilled, spill_tensor


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


def test_release_ops_depend_on_all_consumers() -> None:
    compiled = sc.compile(
        Branching().eval(),
        (torch.randn(2, 16),),
        config=sc.CompileConfig(max_concurrent_regions=2, use_torch_compile=False),
    )
    try:
        schedule = compiled.specialized.schedule
        assert schedule is not None
        releases = [i for i in schedule.instructions if i.opcode == OpCode.RELEASE]
        by_prod = {i.attributes["producer_region"]: i for i in releases}
        # region_0 feeds region_1 and region_2 — both must appear in depends_on.
        rel0 = by_prod["region_0"]
        assert set(rel0.depends_on) == {"compute::region_1", "compute::region_2"}
    finally:
        compiled.close()


def test_schedule_driven_run_releases_and_matches_eager() -> None:
    model = Branching().eval()
    x = torch.randn(2, 16)
    compiled = sc.compile(
        model,
        (x,),
        config=sc.CompileConfig(max_concurrent_regions=2, use_torch_compile=False),
    )
    try:
        assert compiled.executor._schedule_driven
        with torch.no_grad():
            out = compiled(x)
            torch.testing.assert_close(out, model(x), atol=1e-5, rtol=1e-5)
        stats = compiled.last_report.parameter_store if compiled.last_report else {}
        assert stats.get("schedule_driven") is True
        # Intermediate activations should have been released by schedule Release ops.
        assert compiled.last_report is not None
        assert compiled.last_report.released_values >= 1
    finally:
        compiled.close()


def test_activation_allocator_reuses_slot_during_live_run() -> None:
    from streamcompiler.runtime.graph_executor import GraphExecutor
    from streamcompiler.runtime.tensor_store import ResidentParameterStore

    model = Branching().eval()
    x = torch.randn(2, 16)
    compiled = sc.compile(
        model,
        (x,),
        config=sc.CompileConfig(max_concurrent_regions=2, use_torch_compile=False),
    )
    try:
        assignment = compiled.portable.metadata["buffer_reuse"]["assignment"]
        assert assignment.get("submod_0") == assignment.get("submod_3")
        slot = assignment["submod_0"]
        # Physical reuse is only enabled for single-worker runs.
        executor = GraphExecutor(
            compiled.program,
            compiled.executor.bindings,
            parameter_store=ResidentParameterStore(compiled.program.state_tensors()),
            max_workers=1,
            schedule=compiled.specialized.schedule,
            buffer_reuse_assignment=assignment,
        )
        assert executor._allocator is not None
        with torch.no_grad():
            outs, _report = executor.run(compiled.program.flatten_inputs((x,), {}))
            torch.testing.assert_close(outs[0], model(x), atol=1e-5, rtol=1e-5)
        record = executor._allocator.snapshot()[slot]
        assert "submod_0" in record.reuse_history
        assert "submod_3" in record.reuse_history
        assert "reuse" in {e["event"] for e in record.events} or len(record.reuse_history) >= 2
    finally:
        compiled.close()


def test_activation_spill_and_reload_roundtrip() -> None:
    tensor = torch.randn(32, 32)
    spilled = spill_tensor(tensor)
    assert spilled.path.exists()
    restored = reload_spilled(spilled)
    torch.testing.assert_close(restored, tensor)
    assert not spilled.path.exists()


def test_runtime_spills_when_activation_budget_is_tiny() -> None:
    model = Branching().eval()
    x = torch.randn(2, 16)
    # Tiny budget forces spill of intermediate activations during the multi-region run.
    compiled = sc.compile(
        model,
        (x,),
        config=sc.CompileConfig(
            max_concurrent_regions=2,
            use_torch_compile=False,
            activation_budget_bytes=64,
            measure_regions=False,
        ),
    )
    try:
        with torch.no_grad():
            out = compiled(x)
            torch.testing.assert_close(out, model(x), atol=1e-5, rtol=1e-5)
        spills = [e for e in compiled.executor._spill_events if e.get("event") == "spill"]
        assert spills, "expected at least one activation spill under a tiny budget"
        note = " ".join(compiled.specialized.plan.notes)
        assert "runtime disk spill enabled" in note or compiled.executor._allow_activation_spill
    finally:
        compiled.close()
