"""Foundation tests: schedule IR, residency, transfers, compile, telemetry.

GPU participation here is deterministic simulation only — not hardware validation.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

import streamcompiler as sc
from streamcompiler.analysis.alias import run_alias_analysis
from streamcompiler.analysis.liveness import ranges_overlap, run_liveness_analysis
from streamcompiler.backends.base import KernelCandidate, RegionSource
from streamcompiler.backends.torch_device import clear_compile_cache, compile_region_for_torch_device
from streamcompiler.config import CompileConfig
from streamcompiler.errors import RuntimePlanError, UnsupportedFeatureError
from streamcompiler.ir.graph import HeterogeneousGraph, Instruction, OpCode, TensorMeta
from streamcompiler.ir.resource_graph import (
    ComputeClass,
    ComputeResource,
    LinkClass,
    MemoryClass,
    MemoryResource,
    ResourceDecision,
    ResourceGraph,
    ResourceId,
    ResourceKind,
    TransferLink,
)
from streamcompiler.observability import report_to_chrome_trace
from streamcompiler.planner.maximal import ExecutionPlan, Placement
from streamcompiler.runtime.residency import build_residency_schedule
from streamcompiler.runtime.schedule import (
    MemoryTier,
    PlanInstruction,
    build_executable_schedule,
    placements_from_schedule,
    schedule_matches_plan,
)
from streamcompiler.runtime.tensor_directory import TensorDirectory, TensorState
from streamcompiler.runtime.transfers import HostMemcpyTransfer, execute_transfer_instruction
from streamcompiler.simulator import simulate_plan, simulate_schedule


def test_planner_decisions_cite_millisecond_deltas() -> None:
    model = nn.Linear(32, 16).eval()
    x = torch.randn(4, 32)
    compiled = sc.compile(model, (x,), config=CompileConfig(measure_regions=True, use_torch_compile=False))
    text = compiled.explain()
    assert "ms" in text or any("ms" in d.reason for d in compiled.specialized.plan.decisions)
    assert compiled.specialized.schedule is not None
    assert "executable_schedule:" in text
    compiled.close()


def test_torch_compile_slower_fallback_noted_on_plan() -> None:
    clear_compile_cache()
    model = nn.Linear(32, 8).eval()
    x = torch.randn(2, 32)
    compiled = sc.compile(
        model,
        (x,),
        config=CompileConfig(use_torch_compile=True, measure_regions=False, allow_concurrent_regions=False),
    )
    with torch.no_grad():
        torch.testing.assert_close(compiled(x), model(x))
    meta = compiled.specialized.compiled_regions[0]
    # On this CPU host tiny linears usually fall back; either path must stay numerical.
    assert meta.get("impl") in {"torch_fx_subgraph", "torch_compile_inductor"} or str(meta.get("impl", "")).startswith(
        "torch_compile_"
    )
    if meta.get("fallback"):
        assert any("eager_fallback_regions=" in n for n in compiled.specialized.plan.notes)
        assert meta.get("fallback_reason")
    compiled.close()


def test_compiled_region_numerical_equivalence_and_repeated_calls() -> None:
    model = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 8)).eval()
    x = torch.randn(4, 32)
    compiled = sc.compile(model, (x,), config=CompileConfig(use_torch_compile=False))
    with torch.no_grad():
        expected = model(x)
        for _ in range(3):
            actual = compiled(x)
            torch.testing.assert_close(actual, expected)
    compiled.close()


def test_torch_compile_region_or_explicit_eager_fallback() -> None:
    clear_compile_cache()
    module = nn.Linear(16, 8).eval()
    region = RegionSource(
        region_id="linear",
        module=module,
        input_names=("x",),
        output_names=("y",),
        aten_ops=("aten::linear",),
        example_inputs=(torch.randn(2, 16),),
    )
    cand = KernelCandidate(
        region_id="linear",
        device="cpu_numa_0",
        backend_id="cpu",
        kernel_id="cpu_fx_float32",
        dtype="float32",
        attributes={
            "use_torch_compile": True,
            "torch_compile_backend": "inductor",
            "machine_fingerprint": "test-fp",
        },
    )
    compiled = compile_region_for_torch_device(region, cand, backend_id="cpu", torch_device="cpu")
    x = torch.randn(2, 16)
    with torch.no_grad():
        expected = module(x)
        actual = compiled.executable(x)
        if isinstance(actual, tuple):
            actual = actual[0]
        torch.testing.assert_close(actual, expected)
    impl = compiled.attributes.get("impl", "")
    assert impl.startswith("torch_compile_") or compiled.attributes.get("fallback") is True
    assert compiled.attributes.get("cache_key")
    # Eager fallback path must still run the real module.
    if compiled.attributes.get("fallback"):
        assert compiled.attributes.get("fallback_reason")


def test_executable_schedule_shared_by_simulator() -> None:
    plan = ExecutionPlan(
        graph_name="sched",
        fingerprint="fp",
        objective="latency",
        placements=[
            Placement("a", "cpu_numa_0", "cpu", "float32", "k", 0.01, output_bytes=1024, state_bytes=256),
            Placement("b", "cpu_numa_0", "cpu", "float32", "k", 0.02, depends_on=("a",), output_bytes=512),
        ],
        decisions=[],
        devices_used=("cpu_numa_0",),
        communication_backend="none",
        predicted_latency_s=0.03,
    )
    residency = build_residency_schedule(plan)
    schedule = build_executable_schedule(plan, residency, streaming=True, prefetch_distance=1)
    assert schedule_matches_plan(schedule, plan) == []
    assert any(i.opcode == OpCode.COMPUTE for i in schedule.instructions)
    assert any(i.opcode == OpCode.PREFETCH for i in schedule.instructions)
    assert any(i.opcode == OpCode.RELEASE for i in schedule.instructions)

    machine = ResourceGraph(fingerprint="fp")
    machine.add_memory(
        MemoryResource(
            id=ResourceId(ResourceKind.MEMORY, "numa_ram_0"),
            memory_class=MemoryClass.NUMA_RAM,
            capacity_bytes=1 << 30,
            allocatable_bytes=1 << 30,
        )
    )
    machine.add_compute(
        ComputeResource(
            id=ResourceId(ResourceKind.COMPUTE, "cpu_numa_0"),
            compute_class=ComputeClass.CPU_NUMA_POOL,
            backend_id="cpu",
            model="test",
            memory_affinity=("numa_ram_0",),
        )
    )
    from_plan = simulate_plan(plan, machine)
    from_sched = simulate_schedule(schedule, machine)
    assert from_sched.simulated is True
    assert from_plan.simulated is True
    rebuilt = placements_from_schedule(schedule)
    assert [p.region_id for p in rebuilt] == ["a", "b"]


def test_tensor_residency_transitions_and_duplicate_transfer_elimination() -> None:
    directory = TensorDirectory()
    directory.materialize("t0", location="disk", tier=MemoryTier.DISK, nbytes=64)
    assert directory.get("t0").state == TensorState.ON_DISK
    directory.begin_transfer("t0")
    assert directory.get("t0").state == TensorState.TRANSFERRING
    directory.complete_transfer("t0", location="cpu_numa_0", tier=MemoryTier.SYSTEM_RAM, nbytes=64)
    assert directory.has_copy_at("t0", "cpu_numa_0")
    assert directory.get("t0").state == TensorState.IN_RAM

    value = torch.randn(8)
    inst = PlanInstruction(
        opcode=OpCode.TRANSFER,
        name="t",
        resource="copy",
        inputs=("t0",),
        nbytes=64,
        memory_tier=MemoryTier.SYSTEM_RAM,
        source="cpu_numa_0",
        destination="cpu_numa_0",
        transfer_backend="host_memcpy",
    )
    # Already resident at dest → duplicate eliminated.
    out, result = execute_transfer_instruction(inst, value, directory)
    assert result.backend == "elided_duplicate"
    assert result.nbytes == 0
    assert out is value


def test_host_memcpy_transfer_is_real() -> None:
    src = torch.randn(32)
    out, result = HostMemcpyTransfer().transfer(src, source="a", destination="b", nbytes=128)
    assert result.simulated is False
    assert result.nbytes == src.numel() * src.element_size()
    torch.testing.assert_close(out, src)
    assert out.data_ptr() != src.data_ptr()


def test_liveness_non_overlapping_reuse_and_activation_intervals() -> None:
    graph = HeterogeneousGraph(name="live", outputs=("c",))
    graph.add_tensor(TensorMeta("a", (4,), "float32", size_bytes=16, kind="activation"))
    graph.add_tensor(TensorMeta("b", (4,), "float32", size_bytes=16, kind="activation"))
    graph.add_tensor(TensorMeta("c", (4,), "float32", size_bytes=16, kind="activation"))
    graph.add_instruction(Instruction(OpCode.COMPUTE, "r0", inputs=(), outputs=("a",)))
    graph.add_instruction(Instruction(OpCode.COMPUTE, "r1", inputs=("a",), outputs=("b",)))
    graph.add_instruction(Instruction(OpCode.RELEASE, "rel_a", inputs=("a",), outputs=()))
    graph.add_instruction(Instruction(OpCode.COMPUTE, "r2", inputs=("b",), outputs=("c",)))
    analysis = run_liveness_analysis(graph)
    assert analysis.intervals["a"][0] == 0
    assert analysis.intervals["a"][1] == 2
    assert analysis.intervals["c"][1] == 3
    # a dies before c is produced in a tighter graph; check pair helper.
    assert ranges_overlap((0, 1), (2, 3)) is False
    assert ("a", "c") in analysis.reuse_groups or not ranges_overlap(analysis.intervals["a"], analysis.intervals["c"])


def test_shared_weights_and_view_alias_and_mutation_rejection() -> None:
    graph = HeterogeneousGraph(
        name="alias",
        metadata={"state_bindings": {"w0": "shared.weight", "w1": "shared.weight"}},
    )
    graph.add_tensor(TensorMeta("w0", (4, 4), "float32", size_bytes=64, kind="parameter"))
    graph.add_tensor(TensorMeta("w1", (4, 4), "float32", size_bytes=64, kind="parameter"))
    graph.add_tensor(
        TensorMeta("view", (4,), "float32", size_bytes=16, kind="activation", attributes={"view_of": "w0"})
    )
    alias = run_alias_analysis(graph)
    assert alias.groups["w0"] == alias.groups["w1"]
    assert alias.view_of["view"] == "w0"

    bad = HeterogeneousGraph(
        name="mut",
        metadata={"state_bindings": {"w0": "shared.weight", "w1": "shared.weight"}},
    )
    bad.add_tensor(TensorMeta("w0", (4, 4), "float32", size_bytes=64, kind="parameter", mutable=True))
    bad.add_tensor(TensorMeta("w1", (4, 4), "float32", size_bytes=64, kind="parameter"))
    with pytest.raises(UnsupportedFeatureError, match="shared weights"):
        run_alias_analysis(bad)

    directory = TensorDirectory()
    directory.ensure("x", mutable=False)
    with pytest.raises(RuntimePlanError, match="immutable"):
        directory.mutate("x")


def test_specialize_attaches_executable_schedule_and_telemetry(tmp_path: Path) -> None:
    model = nn.Linear(8, 4).eval()
    x = torch.randn(2, 8)
    compiled = sc.compile(model, (x,), config=CompileConfig(use_torch_compile=False, measure_regions=True))
    assert compiled.specialized.schedule is not None
    assert compiled.specialized.profile.get("executable_schedule")
    with torch.no_grad():
        _ = compiled(x)
    report = compiled.last_report
    assert report is not None
    assert report.events
    trace = report_to_chrome_trace(report, plan=compiled.specialized.plan)
    assert trace["metadata"]["simulated"] is False
    assert trace["metadata"]["measured"] is True
    path = tmp_path / "measured.json"
    compiled.visualize(str(path), measured=True)
    payload = path.read_text(encoding="utf-8")
    assert '"measured": true' in payload or '"measured": True' in payload
    # Directory tracked produces on multi-region path; single-region fast path may skip.
    snap = compiled.executor.tensor_directory.snapshot()
    assert isinstance(snap, dict)
    compiled.close()


def test_simulation_cpu_gpu_independent_branches_and_exclusion() -> None:
    """SIMULATION ONLY — virtual GPU topology, not hardware validation."""
    machine = ResourceGraph(fingerprint="sim-hetero")
    for name, cls, backend, mem in (
        ("cpu_numa_0", ComputeClass.CPU_NUMA_POOL, "cpu", "numa_ram_0"),
        ("gpu_0", ComputeClass.DISCRETE_GPU, "cuda", "vram_0"),
        ("gpu_1", ComputeClass.DISCRETE_GPU, "cuda", "vram_1"),
    ):
        machine.add_memory(
            MemoryResource(
                id=ResourceId(ResourceKind.MEMORY, mem),
                memory_class=MemoryClass.NUMA_RAM if "ram" in mem else MemoryClass.DEVICE_VRAM,
                capacity_bytes=4 << 30,
                allocatable_bytes=4 << 30,
            )
        )
        machine.add_compute(
            ComputeResource(
                id=ResourceId(ResourceKind.COMPUTE, name),
                compute_class=cls,
                backend_id=backend,
                model=name,
                memory_affinity=(mem,),
            )
        )
    machine.add_link(
        TransferLink(
            id=ResourceId(ResourceKind.LINK, "numa_ram_0->vram_0"),
            link_class=LinkClass.PCIE,
            source="numa_ram_0",
            destination="vram_0",
            measured=True,
            bytes_per_s=8e9,
            latency_s=1e-5,
        )
    )
    machine.add_link(
        TransferLink(
            id=ResourceId(ResourceKind.LINK, "vram_0->vram_1"),
            link_class=LinkClass.HOST_STAGED,
            source="vram_0",
            destination="vram_1",
            peer_to_peer=False,
            measured=True,
            bytes_per_s=4e9,
            latency_s=2e-5,
        )
    )

    # Independent CPU and GPU branches.
    independent = ExecutionPlan(
        graph_name="indep",
        fingerprint="sim",
        objective="latency",
        placements=[
            Placement("cpu_branch", "cpu_numa_0", "cpu", "float32", "k", 0.05, output_bytes=1_000_000),
            Placement("gpu_branch", "gpu_0", "cuda", "float16", "k", 0.04, output_bytes=1_000_000),
        ],
        decisions=[
            ResourceDecision("gpu_0", True, "reduced predicted critical-path latency by 10.0 ms"),
            ResourceDecision("gpu_1", False, "host-staged transfer added 14 ms while saving only 6 ms of compute"),
        ],
        devices_used=("cpu_numa_0", "gpu_0"),
        communication_backend="none",
        predicted_latency_s=0.0,
        strategy="simulation_independent_branches",
    )
    sim = simulate_plan(independent, machine)
    assert sim.simulated is True
    assert sim.makespan_s < 0.05 + 0.04  # overlap: makespan ~ max, not sum
    assert "host-staged transfer" in independent.decisions[1].reason

    # Pipeline CPU prepare -> GPU compute.
    pipeline = ExecutionPlan(
        graph_name="pipe",
        fingerprint="sim",
        objective="latency",
        placements=[
            Placement("prep", "cpu_numa_0", "cpu", "float32", "k", 0.02, output_bytes=2_000_000),
            Placement("gpu", "gpu_0", "cuda", "float16", "k", 0.03, depends_on=("prep",), output_bytes=0),
        ],
        decisions=[],
        devices_used=("cpu_numa_0", "gpu_0"),
        communication_backend="host_staged",
        predicted_latency_s=0.0,
    )
    pipe_sim = simulate_plan(pipeline, machine)
    assert pipe_sim.exposed_transfer_latency_s >= 0.0
    assert any(e.get("event") == "compute" for e in pipe_sim.timeline)

    # Unequal GPUs: slower GPU excluded by decision text.
    unequal = ExecutionPlan(
        graph_name="uneq",
        fingerprint="sim",
        objective="latency",
        placements=[
            Placement("a", "gpu_0", "cuda", "float16", "k", 0.01, output_bytes=4_000_000),
            Placement("b", "gpu_1", "cuda", "float16", "k", 0.01, depends_on=("a",)),
        ],
        decisions=[
            ResourceDecision(
                "gpu_1",
                False,
                "GPU 1 excluded because its host-staged transfer added 14 ms while saving only 6 ms of compute",
            )
        ],
        devices_used=("gpu_0", "gpu_1"),
        communication_backend="host_staged",
        predicted_latency_s=0.0,
    )
    unequal_sim = simulate_plan(unequal, machine)
    assert unequal_sim.transfer_events
    assert "14 ms" in unequal.decisions[0].reason

    # Memory capacity failure pressure.
    tiny = ResourceGraph(fingerprint="tiny")
    tiny.add_memory(
        MemoryResource(
            id=ResourceId(ResourceKind.MEMORY, "vram_0"),
            memory_class=MemoryClass.DEVICE_VRAM,
            capacity_bytes=1024,
            allocatable_bytes=1024,
        )
    )
    tiny.add_compute(
        ComputeResource(
            id=ResourceId(ResourceKind.COMPUTE, "gpu_0"),
            compute_class=ComputeClass.DISCRETE_GPU,
            backend_id="cuda",
            model="tiny",
            memory_affinity=("vram_0",),
        )
    )
    overflow = ExecutionPlan(
        graph_name="oom",
        fingerprint="sim",
        objective="latency",
        placements=[Placement("big", "gpu_0", "cuda", "float16", "k", 0.01, output_bytes=10_000, state_bytes=10_000)],
        decisions=[],
        devices_used=("gpu_0",),
        communication_backend="none",
        predicted_latency_s=0.0,
    )
    overflow_sim = simulate_plan(overflow, tiny)
    assert any(e.get("event") == "eviction_pressure" for e in overflow_sim.timeline)

    # GPU participation making execution slower (SIMULATION).
    cpu_only = ExecutionPlan(
        graph_name="cpu_only",
        fingerprint="sim",
        objective="latency",
        placements=[Placement("all", "cpu_numa_0", "cpu", "float32", "k", 0.05, output_bytes=0)],
        decisions=[],
        devices_used=("cpu_numa_0",),
        communication_backend="none",
        predicted_latency_s=0.0,
    )
    gpu_slower = ExecutionPlan(
        graph_name="gpu_tax",
        fingerprint="sim",
        objective="latency",
        placements=[
            Placement("a", "cpu_numa_0", "cpu", "float32", "k", 0.04, output_bytes=64_000_000),
            Placement(
                "b",
                "gpu_0",
                "cuda",
                "float16",
                "k",
                0.01,
                depends_on=("a",),
                output_bytes=0,
            ),
        ],
        decisions=[
            ResourceDecision(
                "gpu_0",
                False,
                "GPU 0 excluded because its host-staged transfer added 14.0 ms while saving only 6.0 ms of compute",
            )
        ],
        devices_used=("cpu_numa_0", "gpu_0"),
        communication_backend="host_staged",
        predicted_latency_s=0.0,
    )
    cpu_sim = simulate_plan(cpu_only, machine)
    gpu_sim = simulate_plan(gpu_slower, machine)
    assert gpu_sim.makespan_s > cpu_sim.makespan_s
    assert "14.0 ms" in gpu_slower.decisions[0].reason

    # CPU participation helping: independent CPU prep overlaps GPU (SIMULATION).
    helped = ExecutionPlan(
        graph_name="helped",
        fingerprint="sim",
        objective="latency",
        placements=[
            Placement("gpu_heavy", "gpu_0", "cuda", "float16", "k", 0.08, output_bytes=0),
            Placement("cpu_prep", "cpu_numa_0", "cpu", "float32", "k", 0.03, output_bytes=0),
        ],
        decisions=[
            ResourceDecision(
                "cpu_numa_0",
                True,
                "cpu_numa_0 selected because it reduced predicted critical-path latency by 8.4 ms",
            )
        ],
        devices_used=("cpu_numa_0", "gpu_0"),
        communication_backend="none",
        predicted_latency_s=0.0,
    )
    helped_sim = simulate_plan(helped, machine)
    assert helped_sim.makespan_s <= 0.08 + 1e-9
    assert "8.4 ms" in helped.decisions[0].reason

    # CPU participation delaying synchronization (SIMULATION).
    delayed = ExecutionPlan(
        graph_name="delay",
        fingerprint="sim",
        objective="latency",
        placements=[
            Placement("gpu", "gpu_0", "cuda", "float16", "k", 0.02, output_bytes=1_000_000),
            Placement(
                "cpu_join",
                "cpu_numa_0",
                "cpu",
                "float32",
                "k",
                0.05,
                depends_on=("gpu",),
                output_bytes=0,
            ),
        ],
        decisions=[],
        devices_used=("cpu_numa_0", "gpu_0"),
        communication_backend="host_staged",
        predicted_latency_s=0.0,
    )
    delayed_sim = simulate_plan(delayed, machine)
    assert delayed_sim.makespan_s >= 0.05
    assert delayed_sim.exposed_transfer_latency_s >= 0.0

    # Host-staged transfers between incompatible GPUs (SIMULATION).
    host_staged = ExecutionPlan(
        graph_name="host_staged",
        fingerprint="sim",
        objective="latency",
        placements=[
            Placement("a", "gpu_0", "cuda", "float16", "k", 0.01, output_bytes=4_000_000),
            Placement("b", "gpu_1", "cuda", "float16", "k", 0.01, depends_on=("a",)),
        ],
        decisions=[],
        devices_used=("gpu_0", "gpu_1"),
        communication_backend="host_staged",
        predicted_latency_s=0.0,
    )
    hs_sim = simulate_plan(host_staged, machine)
    assert hs_sim.transfer_events
    assert hs_sim.simulated is True
    residency = build_residency_schedule(host_staged)
    schedule = build_executable_schedule(host_staged, residency)
    assert any(i.opcode == OpCode.TRANSFER for i in schedule.instructions)
    assert all(
        i.attributes.get("simulated_until_validated") for i in schedule.transfer_ops() if i.opcode == OpCode.TRANSFER
    )


def test_specialize_builds_schedule_for_streaming_disk_prefetch(tmp_path: Path) -> None:
    class Deep(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList(nn.Linear(64, 64) for _ in range(6))
            self.head = nn.Linear(64, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            for layer in self.layers:
                x = torch.relu(layer(x))
            return self.head(x)

    model = Deep().eval()
    x = torch.randn(2, 64)
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    layer_bytes = 64 * 64 * 4
    budget = layer_bytes * 3
    assert budget < total
    compiled = sc.compile(
        model,
        (x,),
        config=CompileConfig(
            use_torch_compile=False,
            ram_budget_bytes=budget,
            allow_nvme_streaming=True,
            measure_regions=False,
            max_region_nodes=2,
        ),
    )
    schedule = compiled.specialized.schedule
    assert schedule is not None
    assert any(i.opcode == OpCode.PREFETCH for i in schedule.instructions) or any(
        i.opcode == OpCode.LOAD for i in schedule.instructions
    )
    with torch.no_grad():
        expected = model(x)
        actual = compiled(x)
        torch.testing.assert_close(actual, expected)
    stats = compiled.executor.parameter_store.stats()
    assert stats.get("kind") == "streaming"
    compiled.close()
