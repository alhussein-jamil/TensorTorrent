"""Native planner + batch DES architecture tests."""

from __future__ import annotations

import torch
import torch.nn as nn

import tensortorrent as tt
from tensortorrent.config import CompileConfig
from tensortorrent.ir.graph import OpCode
from tensortorrent.ir.resource_graph import (
    ComputeClass,
    ComputeResource,
    MemoryClass,
    MemoryResource,
    ResourceGraph,
    ResourceId,
    ResourceKind,
)
from tensortorrent.native import require_native
from tensortorrent.runtime.schedule import ExecutableSchedule, PlanInstruction
from tensortorrent.runtime.simulator.discrete_event import simulate_schedule, simulate_schedules


def _tiny_machine() -> ResourceGraph:
    machine = ResourceGraph(fingerprint="native-planner-test", backends_present=("cpu",))
    machine.add_memory(
        MemoryResource(
            id=ResourceId(ResourceKind.MEMORY, "host_ram"),
            memory_class=MemoryClass.NUMA_RAM,
            capacity_bytes=1 << 30,
            allocatable_bytes=1 << 30,
            attached_compute=("cpu0",),
        )
    )
    machine.add_compute(
        ComputeResource(
            id=ResourceId(ResourceKind.COMPUTE, "cpu0"),
            compute_class=ComputeClass.CPU_NUMA_POOL,
            backend_id="cpu",
            vendor="test",
            model="cpu0",
            memory_affinity=("host_ram",),
            supported_dtypes=("float32",),
        )
    )
    return machine


def test_batch_sim_matches_scalar_and_preserves_order() -> None:
    machine = _tiny_machine()
    s1 = ExecutableSchedule(
        graph_name="g",
        fingerprint="fp",
        instructions=(
            PlanInstruction(
                opcode=OpCode.COMPUTE,
                name="c0",
                resource="cpu0",
                outputs=("o0",),
                nbytes=64,
                predicted_duration_s=0.05,
            ),
        ),
    )
    s2 = ExecutableSchedule(
        graph_name="g",
        fingerprint="fp",
        instructions=(
            PlanInstruction(
                opcode=OpCode.COMPUTE,
                name="c1",
                resource="cpu0",
                outputs=("o1",),
                nbytes=64,
                predicted_duration_s=0.2,
            ),
        ),
    )
    r1 = simulate_schedule(s1, machine)
    r2 = simulate_schedule(s2, machine)
    batch = simulate_schedules([s1, s2], machine, workers=1)
    assert len(batch) == 2
    assert abs(batch[0].makespan_s - r1.makespan_s) < 1e-15
    assert abs(batch[1].makespan_s - r2.makespan_s) < 1e-15
    assert batch[0].initiation_interval_s > 0
    parallel = simulate_schedules([s1, s2, s1], machine, workers=4)
    serial = simulate_schedules([s1, s2, s1], machine, workers=1)
    assert [x.makespan_s for x in parallel] == [x.makespan_s for x in serial]


def test_specialize_exposes_rust_planner_stats() -> None:
    class _Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.a = nn.Linear(16, 16)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.a(x)

    model = _Tiny().eval()
    x = torch.randn(2, 16)
    compiled = tt.compile(
        model,
        (x,),
        config=CompileConfig(
            allow_gpu=False,
            use_torch_compile=False,
            measure_regions=False,
            planner_workers=1,
            planner_des_candidates=4,
        ),
    )
    try:
        stats = compiled.specialized.plan.search_statistics
        assert stats.get("planner_engine") == "rust"
        assert compiled.specialized.validation.get("regions_compiled", 0) >= 1
    finally:
        compiled.close()


def test_native_bindings_present() -> None:
    native = require_native()
    assert callable(native.plan_placements)
    assert callable(native.simulate_schedules)
