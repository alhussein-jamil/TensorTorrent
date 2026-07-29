"""AllocationTable wiring, portable homes, ordered streams, sim CP edges."""

from __future__ import annotations

import torch
import torch.nn as nn

import streamcompiler as sc
from streamcompiler.backends.base import RegionSource
from streamcompiler.backends.torch_device import region_compile_fingerprint
from streamcompiler.config import CompileConfig
from streamcompiler.frontend.lower import ir_from_region_program
from streamcompiler.ir.graph import OpCode
from streamcompiler.ir.resource_graph import (
    ComputeClass,
    ComputeResource,
    MemoryClass,
    MemoryResource,
    ResourceGraph,
    ResourceId,
    ResourceKind,
)
from streamcompiler.runtime.copies import CopyStore
from streamcompiler.runtime.execution_context import ExecutionContext
from streamcompiler.runtime.schedule import ExecutableSchedule, PlanInstruction
from streamcompiler.runtime.streams import MockStream
from streamcompiler.simulator.discrete_event import simulate_schedule


def test_alias_shares_one_physical_allocation() -> None:
    ctx = ExecutionContext()
    t = torch.randn(32)
    ctx.copies.put("a", "cpu", t)
    ctx.copies.alias("a", "cpu", "host")
    assert ctx.allocations.live_bytes() == t.numel() * t.element_size()
    assert ctx.copies.live_bytes() == ctx.allocations.live_bytes()
    assert ctx.copies.drop("a", "host") == 0  # ref remains
    assert ctx.allocations.live_bytes() == t.numel() * t.element_size()
    assert ctx.copies.drop("a", "cpu") == t.numel() * t.element_size()
    assert ctx.allocations.live_bytes() == 0


def test_unbound_copy_store_alias_counted_once() -> None:
    store = CopyStore()
    t = torch.ones(8)
    store.put("x", "cpu", t)
    store.alias("x", "cpu", "host")
    assert store.live_bytes() == t.numel() * t.element_size()
    store.drop("x", "host")
    assert store.live_bytes() == t.numel() * t.element_size()
    store.drop("x", "cpu")
    assert store.live_bytes() == 0


def test_portable_ir_has_no_machine_resource_ids() -> None:
    model = nn.Linear(4, 4).eval()
    x = torch.randn(2, 4)
    compiled = sc.compile(model, (x,), config=CompileConfig(use_torch_compile=False, measure_regions=False))
    try:
        graph = ir_from_region_program(compiled.program)
        forbidden = ("numa_ram_", "cuda:", "mock_accel_", "cpu_numa_")
        for tensor in graph.tensors.values():
            home = str(tensor.home_tier or "")
            assert home in {"parameter_home", "host_memory", "unassigned", "persistent_storage", ""}
            assert not any(tok in home for tok in forbidden)
        for tensor in compiled.portable.ir.tensors.values():
            home = str(tensor.home_tier or "")
            assert not any(tok in home for tok in forbidden)
    finally:
        compiled.close()


def test_mock_stream_preserves_submission_order() -> None:
    stream = MockStream("ordered", delay_s=0.01, workers=4)
    try:
        order: list[int] = []

        def _job(i: int) -> int:
            order.append(i)
            return i

        futs = [stream.submit(_job, i) for i in range(5)]
        assert [f.result() for f in futs] == list(range(5))
        assert order == list(range(5))
    finally:
        stream.shutdown()


def test_sim_critical_path_includes_shared_compute_stream() -> None:
    machine = ResourceGraph(fingerprint="sim-cp")
    machine.add_memory(
        MemoryResource(
            id=ResourceId(ResourceKind.MEMORY, "host_ram"),
            memory_class=MemoryClass.NUMA_RAM,
            capacity_bytes=1 << 30,
            allocatable_bytes=1 << 30,
        )
    )
    machine.add_compute(
        ComputeResource(
            id=ResourceId(ResourceKind.COMPUTE, "cpu"),
            compute_class=ComputeClass.CPU_SOCKET,
            backend_id="cpu",
            model="test",
            memory_affinity=("host_ram",),
        )
    )
    # Two independent computes (no explicit deps) on the same resource.
    a = PlanInstruction(
        opcode=OpCode.COMPUTE,
        name="c0",
        resource="cpu",
        outputs=("o0",),
        nbytes=64,
        predicted_duration_s=0.1,
        attributes={"mock_compute_delay_s": 0.1},
    )
    b = PlanInstruction(
        opcode=OpCode.COMPUTE,
        name="c1",
        resource="cpu",
        outputs=("o1",),
        nbytes=64,
        predicted_duration_s=0.1,
        attributes={"mock_compute_delay_s": 0.1},
    )
    schedule = ExecutableSchedule(graph_name="shared_stream", fingerprint="t", instructions=(a, b))
    result = simulate_schedule(schedule, machine)
    assert result.makespan_s >= 0.19
    assert set(result.critical_path) == {"c0", "c1"}
    assert result.simulated is True


def test_allocation_peak_matches_report_after_run() -> None:
    model = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 4)).eval()
    x = torch.randn(2, 16)
    compiled = sc.compile(
        model,
        (x,),
        config=CompileConfig(use_torch_compile=False, measure_regions=False, activation_budget_bytes=1 << 20),
    )
    try:
        y = compiled(x)
        assert y.shape == (2, 4)
        assert compiled.last_report is not None
        assert compiled.last_report.allocation_peak_bytes > 0
        assert compiled.last_report.peak_activation_bytes > 0
    finally:
        compiled.close()


def test_runtime_activation_budget_rejects_durable_overage() -> None:
    """Budget must hard-fail when spillable activations stay over budget with no pending spill."""
    from streamcompiler.errors import RuntimePlanError

    class Branch(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.stem = nn.Linear(16, 16)
            self.left = nn.Linear(16, 8)
            self.right = nn.Linear(16, 8)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.stem(x)
            return self.left(h) + self.right(h)

    model = Branch().eval()
    x = torch.randn(2, 16)
    compiled = sc.compile(
        model,
        (x,),
        config=CompileConfig(
            use_torch_compile=False,
            measure_regions=False,
            max_concurrent_regions=2,
            activation_budget_bytes=1 << 20,
        ),
    )
    try:
        # Force a tiny budget after planning so the schedule has no matching spills.
        compiled.executor.activation_budget_bytes = 1
        sexec = getattr(compiled.executor, "_schedule_executor", None)
        assert sexec is not None
        sexec.activation_budget_bytes = 1
        with torch.no_grad():
            try:
                compiled(x)
                raised = False
            except RuntimePlanError as exc:
                raised = True
                assert "activation budget" in str(exc)
                assert "exceeded" in str(exc)
        assert raised, "expected RuntimePlanError when durable residency exceeds budget"
    finally:
        compiled.close()


def test_runtime_activation_budget_allows_protected_outputs() -> None:
    """End-of-run outputs may exceed budget alone when no spillable leftovers remain."""
    model = nn.Linear(8, 8).eval()
    x = torch.randn(2, 8)
    out_bytes = 2 * 8 * 4
    compiled = sc.compile(
        model,
        (x,),
        config=CompileConfig(
            use_torch_compile=False,
            measure_regions=False,
            activation_budget_bytes=max(1, out_bytes // 2),
        ),
    )
    try:
        with torch.no_grad():
            y = compiled(x)
        assert y.shape == (2, 8)
        assert compiled.last_report is not None
        assert compiled.last_report.peak_activation_bytes >= 0
    finally:
        compiled.close()


def test_device_load_prefers_pinned_host_staging() -> None:
    from streamcompiler.planner.maximal import ExecutionPlan, Placement
    from streamcompiler.runtime.schedule import MemoryTier, build_executable_schedule

    machine = ResourceGraph(fingerprint="pinned-stage")
    machine.add_memory(
        MemoryResource(
            id=ResourceId(ResourceKind.MEMORY, "pinned_host_0"),
            memory_class=MemoryClass.PINNED_HOST,
            capacity_bytes=1 << 30,
            allocatable_bytes=1 << 30,
        )
    )
    machine.add_memory(
        MemoryResource(
            id=ResourceId(ResourceKind.MEMORY, "mock_vram_0"),
            memory_class=MemoryClass.DEVICE_VRAM,
            capacity_bytes=1 << 30,
            allocatable_bytes=1 << 30,
        )
    )
    machine.add_compute(
        ComputeResource(
            id=ResourceId(ResourceKind.COMPUTE, "mock_accel_0"),
            compute_class=ComputeClass.ACCELERATOR,
            backend_id="mock_accel",
            model="test",
            memory_affinity=("mock_vram_0",),
        )
    )
    plan = ExecutionPlan(
        graph_name="pinned",
        fingerprint="t",
        objective="latency",
        placements=[
            Placement(
                region_id="region_0",
                device="mock_accel_0",
                backend_id="mock_accel",
                dtype="float32",
                kernel_id="k",
                estimated_latency_s=0.01,
                state_bytes=1024,
                output_bytes=64,
            )
        ],
        decisions=[],
        devices_used=("mock_accel_0",),
        communication_backend="none",
        predicted_latency_s=0.01,
    )
    schedule = build_executable_schedule(plan, streaming=True, machine=machine)
    loads = [
        i
        for i in schedule.instructions
        if i.opcode == OpCode.LOAD and i.attributes.get("kind") == "parameter_materialize"
    ]
    assert loads
    assert loads[0].destination == "pinned_host_0"
    assert loads[0].memory_tier == MemoryTier.PINNED_RAM
    xfers = [
        i
        for i in schedule.instructions
        if i.opcode == OpCode.TRANSFER and i.attributes.get("kind") == "parameter_host_to_device"
    ]
    assert xfers
    assert xfers[0].source == "pinned_host_0"
    assert xfers[0].transfer_backend == "host_device_copy"


def test_region_module_rejects_undeclared_parameters() -> None:
    from streamcompiler.backends.base import KernelCandidate, RegionSource
    from streamcompiler.backends.torch_device import compile_region_for_torch_device
    from streamcompiler.errors import BackendError

    class _Stateful(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = nn.Parameter(torch.ones(2, 2))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x @ self.w

    region = RegionSource(
        region_id="bad",
        module=_Stateful().eval(),
        input_names=("x",),
        output_names=("y",),
        aten_ops=("aten.mm.default",),
        example_inputs=(torch.randn(2, 2),),
        attributes={"declared_state": ()},
    )
    candidate = KernelCandidate(
        region_id="bad",
        device="cpu",
        backend_id="cpu",
        kernel_id="eager",
        dtype="float32",
        attributes={"use_torch_compile": False, "schedule_managed_placement": True},
    )
    try:
        compile_region_for_torch_device(region, candidate, backend_id="cpu", torch_device="cpu")
        raised = False
    except BackendError as exc:
        raised = True
        assert "unexpected_parameters" in str(exc)
    assert raised


def test_pack_tensors_invokes_loaders_one_at_a_time(tmp_path) -> None:
    from streamcompiler.storage.pack import load_pack_manifest, pack_tensors

    live: list[str] = []
    peak = 0

    def _loader(name: str, n: int):
        def _load() -> torch.Tensor:
            live.append(name)
            nonlocal peak
            peak = max(peak, len(live))
            try:
                return torch.ones(n)
            finally:
                live.remove(name)

        return _load

    pack = pack_tensors(
        (("a", _loader("a", 64)), ("b", _loader("b", 128)), ("c", _loader("c", 32))),
        tmp_path / "lazy.pack",
    )
    assert peak == 1
    manifest = load_pack_manifest(pack.path)
    assert [t["logical_id"] for t in manifest["tensors"]] == ["a", "b", "c"]


def test_inductor_fingerprint_changes_with_shape() -> None:
    class _M(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + 1

    mod = _M().eval()
    region = RegionSource(
        region_id="r0",
        module=mod,
        aten_ops=("aten.add.Tensor",),
        input_names=("x",),
        output_names=("y",),
        example_inputs=(torch.randn(2, 4),),
    )
    k1 = region_compile_fingerprint(
        region,
        torch_device="cpu",
        backend="inductor",
        dtype="float32",
        input_shapes=((2, 4),),
        input_dtypes=("float32",),
        input_strides=((4, 1),),
        input_layouts=("contiguous",),
    )
    k2 = region_compile_fingerprint(
        region,
        torch_device="cpu",
        backend="inductor",
        dtype="float32",
        input_shapes=((4, 4),),
        input_dtypes=("float32",),
        input_strides=((4, 1),),
        input_layouts=("contiguous",),
    )
    assert k1 != k2
