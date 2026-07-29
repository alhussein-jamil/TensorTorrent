"""Host-backed mock accelerator for heterogeneous schedule tests.

Not discovered by default (``available()`` is False). Tests inject a
``mock_accel_0`` resource into a ``ResourceGraph`` and compile through this
backend so CPU+accelerator schedules can be exercised without a GPU.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from streamcompiler.backends.base import (
    BenchmarkResult,
    CompiledRegion,
    ExecutionBackend,
    KernelCandidate,
    RegionSource,
    TransferCapability,
    region_identifier,
)
from streamcompiler.backends.torch_device import (
    compile_region_for_torch_device,
    execute_region_on_torch_device,
)
from streamcompiler.ir.graph import HeterogeneousGraph, Instruction
from streamcompiler.ir.resource_graph import (
    ComputeClass,
    ComputeResource,
    LinkClass,
    MemoryClass,
    MemoryResource,
    ResourceGraph,
    ResourceId,
    ResourceKind,
    TransferLink,
)


class MockAccelBackend(ExecutionBackend):
    backend_id = "mock_accel"

    def available(self) -> bool:
        # Never auto-discovered; tests inject resources explicitly.
        return False

    def discover_devices(self) -> ResourceGraph:
        return make_mock_accel_graph()

    def supported_ops(self, device: ComputeResource) -> tuple[str, ...]:
        return ("aten::*",)

    def supported_dtypes(self, device: ComputeResource) -> tuple[str, ...]:
        return ("float32", "float16", "bfloat16")

    def enumerate_kernels(
        self, region: Instruction | HeterogeneousGraph, device: ComputeResource
    ) -> list[KernelCandidate]:
        rid = region_identifier(region)
        return [
            KernelCandidate(
                region_id=rid,
                device=device.id.name,
                backend_id=self.backend_id,
                kernel_id="mock_accel_f32",
                dtype="float32",
                estimated_latency_s=0.01,
                attributes={"mock": True},
            )
        ]

    def benchmark(self, candidate: KernelCandidate) -> BenchmarkResult:
        return BenchmarkResult(
            candidate=candidate,
            latency_s=0.01,
            memory_bytes=0,
            measured=False,
            notes="mock accelerator analytic prior",
        )

    def compile(self, region: RegionSource, candidate: KernelCandidate) -> CompiledRegion:
        delay = float(candidate.attributes.get("mock_delay_s", 0.05))
        attrs = dict(candidate.attributes)
        attrs["schedule_managed_placement"] = True
        cand = KernelCandidate(
            region_id=candidate.region_id,
            device=candidate.device,
            backend_id=candidate.backend_id,
            kernel_id=candidate.kernel_id,
            dtype=candidate.dtype,
            estimated_latency_s=candidate.estimated_latency_s,
            workspace_bytes=candidate.workspace_bytes,
            attributes=attrs,
        )
        compiled = compile_region_for_torch_device(
            region,
            cand,
            backend_id=self.backend_id,
            torch_device="cpu",
        )
        # Delay lives on DeviceStreams / schedule attrs — not sleep() in the
        # calling thread. Executable stays the real FX/Inductor callable.
        compiled.torch_device = "cpu"
        compiled.attributes["mock_delay_s"] = delay
        compiled.attributes["async_stream_delay"] = True
        return compiled

    def execute(self, executable: CompiledRegion, inputs: Sequence[Any]) -> tuple[Any, ...]:
        return execute_region_on_torch_device(executable, inputs)

    def transfer_capabilities(
        self, source: ComputeResource | str, destination: ComputeResource | str
    ) -> TransferCapability:
        src = source if isinstance(source, str) else source.id.name
        dst = destination if isinstance(destination, str) else destination.id.name
        return TransferCapability(src, dst, kind="dma", notes="mock host-staged DMA")

    def resource_to_torch_device(self, resource_id: str) -> Any:
        import torch

        # Host-backed simulation: tensors stay on CPU; residency is schedule-logical.
        return torch.device("cpu")


def make_mock_accel_graph(
    *,
    delay_hint_s: float = 0.05,
    device_count: int = 1,
    capacities_bytes: tuple[int, ...] | None = None,
    delay_hints_s: tuple[float, ...] | None = None,
) -> ResourceGraph:
    """Mock accelerator(s) + VRAM with host↔device links for schedule tests.

    ``device_count>1`` builds unequal multi-device topologies (no P2P — host-staged).
    """
    count = max(1, int(device_count))
    caps = capacities_bytes or tuple(8 * (1 << 30) for _ in range(count))
    delays = delay_hints_s or tuple(float(delay_hint_s) for _ in range(count))
    if len(caps) < count or len(delays) < count:
        raise ValueError("capacities_bytes/delay_hints_s must cover device_count")
    graph = ResourceGraph(fingerprint=f"mock-accel-x{count}", backends_present=("mock_accel",))
    for i in range(count):
        name = f"mock_accel_{i}"
        vram = f"mock_vram_{i}"
        graph.add_memory(
            MemoryResource(
                id=ResourceId(ResourceKind.MEMORY, vram),
                memory_class=MemoryClass.DEVICE_VRAM,
                capacity_bytes=int(caps[i]),
                allocatable_bytes=max(1, int(caps[i]) * 7 // 8),
                attached_compute=(name,),
            )
        )
        graph.add_compute(
            ComputeResource(
                id=ResourceId(ResourceKind.COMPUTE, name),
                compute_class=ComputeClass.ACCELERATOR,
                backend_id="mock_accel",
                vendor="mock",
                model=f"mock-accel-{i}",
                memory_affinity=(vram,),
                supported_dtypes=("float32", "float16", "bfloat16"),
                attributes={"mock_delay_s": float(delays[i]), "fingerprint": f"mock-accel-{i}"},
            )
        )
        graph.add_link(
            TransferLink(
                id=ResourceId(ResourceKind.LINK, f"host->{vram}"),
                link_class=LinkClass.PCIE,
                source="host",
                destination=vram,
                bidirectional=True,
                peer_to_peer=False,
                measured=False,
                latency_s=5e-6,
                bytes_per_s=16e9,
            )
        )
    return graph
