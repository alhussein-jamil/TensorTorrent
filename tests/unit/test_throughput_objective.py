"""Throughput objective must prefer the lower-makespan plan."""

from __future__ import annotations

from tensortorrent.compile.measure import MeasurementSet, RegionMeasurement
from tensortorrent.config import CompileConfig, Objective
from tensortorrent.ir.graph import HeterogeneousGraph, Instruction, OpCode
from tensortorrent.ir.resource_graph import (
    ComputeClass,
    ComputeResource,
    MemoryClass,
    MemoryResource,
    ResourceGraph,
    ResourceId,
    ResourceKind,
)
from tensortorrent.planner.maximal import _score_plan, plan_execution


def _two_cpu_machine() -> ResourceGraph:
    g = ResourceGraph(fingerprint="tput-test", backends_present=("cpu",))
    for i in (0, 1):
        mname = f"numa_ram_{i}"
        cname = f"cpu_numa_{i}"
        g.add_memory(
            MemoryResource(
                id=ResourceId(ResourceKind.MEMORY, mname),
                memory_class=MemoryClass.NUMA_RAM,
                capacity_bytes=32 << 30,
                allocatable_bytes=30 << 30,
                numa_node=i,
            )
        )
        g.add_compute(
            ComputeResource(
                id=ResourceId(ResourceKind.COMPUTE, cname),
                compute_class=ComputeClass.CPU_NUMA_POOL,
                backend_id="cpu",
                model=f"cpu-{i}",
                vendor="cpu",
                supported_dtypes=("float32",),
                supported_ops=("aten::mm",),
                core_count=8,
                concurrency_limit=8,
                numa_node=i,
                memory_affinity=(mname,),
            )
        )
    return g


def test_throughput_score_is_lower_for_faster_plans() -> None:
    """Lower score must mean better under the planner's minimization."""
    cfg = CompileConfig(objective=Objective.THROUGHPUT)
    fast = _score_plan(0.01, cfg)
    slow = _score_plan(0.10, cfg)
    assert fast < slow, f"throughput score inverted: fast={fast} slow={slow}"


def test_memory_objective_prefers_smaller_working_set() -> None:
    cfg = CompileConfig(objective=Objective.MEMORY)
    small = _score_plan(0.05, cfg, peak_working_set_bytes=1000)
    large = _score_plan(0.01, cfg, peak_working_set_bytes=10_000)
    assert small < large


def test_peak_working_set_is_per_device_max() -> None:
    from tensortorrent.planner.maximal import Placement, _peak_working_set_bytes

    placements = [
        Placement("a", "cpu_0", "cpu", "float32", "k", 0.1, output_bytes=100, state_bytes=50),
        Placement("b", "cpu_0", "cpu", "float32", "k", 0.1, output_bytes=20, state_bytes=10),
        Placement("c", "cpu_1", "cpu", "float32", "k", 0.1, output_bytes=40, state_bytes=10),
    ]
    # cpu_0 contributes max(150, 30)=150; cpu_1 contributes 50.
    assert _peak_working_set_bytes(placements) == 200


def test_throughput_objective_selects_faster_measured_device() -> None:
    ir = HeterogeneousGraph(name="tput")
    ir.add_instruction(Instruction(opcode=OpCode.COMPUTE, name="region_0", outputs=()))
    measurements = MeasurementSet()
    measurements.add(
        RegionMeasurement(
            region_id="region_0",
            device="cpu_numa_0",
            backend_id="cpu",
            latency_s=0.10,
            measured=True,
            notes="slow",
        )
    )
    measurements.add(
        RegionMeasurement(
            region_id="region_0",
            device="cpu_numa_1",
            backend_id="cpu",
            latency_s=0.01,
            measured=True,
            notes="fast",
        )
    )
    plan = plan_execution(
        ir,
        _two_cpu_machine(),
        CompileConfig(objective=Objective.THROUGHPUT, allow_gpu=False, max_plan_candidates=16),
        measurements,
    )
    assert plan.devices_used == ("cpu_numa_1",), plan.devices_used
    assert plan.predicted_latency_s == 0.01
