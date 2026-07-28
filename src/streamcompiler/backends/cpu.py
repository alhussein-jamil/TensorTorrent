"""CPU execution backend.

Always available when PyTorch is installed. Discovers sockets/NUMA pools as
independent compute resources rather than one homogenized CPU.
"""

from __future__ import annotations

import os
import platform
import time
from collections.abc import Sequence
from typing import Any

import psutil

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
    benchmark_region_on_torch_device,
    compile_region_for_torch_device,
    execute_region_on_torch_device,
)
from streamcompiler.hardware.topology import read_lscpu_topology
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


def _cpu_vector_isas() -> tuple[str, ...]:
    flags: set[str] = set()
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("flags") or line.startswith("Features"):
                    flags.update(line.split(":", 1)[1].split())
    except OSError:
        pass
    interesting = (
        "avx",
        "avx2",
        "avx512f",
        "avx512_bf16",
        "avx512_fp16",
        "amx_bf16",
        "amx_int8",
        "neon",
        "sve",
    )
    return tuple(sorted(f for f in interesting if f in flags))


def _numa_nodes() -> list[int]:
    base = "/sys/devices/system/node"
    if not os.path.isdir(base):
        return [0]
    nodes = []
    for name in sorted(os.listdir(base)):
        if name.startswith("node") and name[4:].isdigit():
            nodes.append(int(name[4:]))
    return nodes or [0]


class CpuBackend(ExecutionBackend):
    backend_id = "cpu"

    def available(self) -> bool:
        return True

    def discover_devices(self) -> ResourceGraph:
        graph = ResourceGraph(fingerprint="", backends_present=(self.backend_id,))
        isas = _cpu_vector_isas()
        topo = read_lscpu_topology()
        numa_nodes = topo.numa_nodes or _numa_nodes()
        vm = psutil.virtual_memory()
        socket_count = max(1, topo.sockets)
        logical = psutil.cpu_count(logical=True) or 1
        physical = psutil.cpu_count(logical=False) or logical
        per_node_cores = max(1, physical // max(1, len(numa_nodes)))
        per_node_logical = max(1, logical // max(1, len(numa_nodes)))
        mem_per_node = vm.total // max(1, len(numa_nodes))

        for node in numa_nodes:
            mem_name = f"numa_ram_{node}"
            graph.add_memory(
                MemoryResource(
                    id=ResourceId(ResourceKind.MEMORY, mem_name),
                    memory_class=MemoryClass.NUMA_RAM,
                    capacity_bytes=mem_per_node,
                    allocatable_bytes=int(mem_per_node * 0.9),
                    numa_node=node,
                    attributes={"source": "psutil/sysfs"},
                )
            )
            pinned_name = f"pinned_host_{node}"
            graph.add_memory(
                MemoryResource(
                    id=ResourceId(ResourceKind.MEMORY, pinned_name),
                    memory_class=MemoryClass.PINNED_HOST,
                    capacity_bytes=int(mem_per_node * 0.25),
                    allocatable_bytes=int(mem_per_node * 0.2),
                    numa_node=node,
                )
            )
            cpu_name = f"cpu_numa_{node}"
            graph.add_compute(
                ComputeResource(
                    id=ResourceId(ResourceKind.COMPUTE, cpu_name),
                    compute_class=ComputeClass.CPU_NUMA_POOL,
                    backend_id=self.backend_id,
                    model=platform.processor() or platform.machine(),
                    architecture=platform.machine(),
                    vendor=platform.system(),
                    supported_dtypes=self._dtype_probe(),
                    supported_ops=self._op_probe(),
                    core_count=per_node_cores,
                    vector_isas=isas,
                    concurrency_limit=per_node_logical,
                    numa_node=node,
                    memory_affinity=(mem_name, pinned_name),
                    attributes={
                        "logical_cpus": per_node_logical,
                        "sockets_hint": socket_count,
                        "cores_per_socket": topo.cores_per_socket,
                        "threads_per_core": topo.threads_per_core,
                        "intraop_threads": per_node_logical,
                        "fingerprint": f"cpu-numa-{node}-cores{per_node_cores}-log{per_node_logical}",
                    },
                )
            )
            graph.add_link(
                TransferLink(
                    id=ResourceId(ResourceKind.LINK, f"{cpu_name}->{mem_name}"),
                    link_class=LinkClass.CPU_LOCAL,
                    source=cpu_name,
                    destination=mem_name,
                    bidirectional=True,
                    peer_to_peer=True,
                    measured=False,
                )
            )

        # Cross-NUMA interconnects (independent of GPU links).
        for a in numa_nodes:
            for b in numa_nodes:
                if a == b:
                    continue
                graph.add_link(
                    TransferLink(
                        id=ResourceId(ResourceKind.LINK, f"numa_ram_{a}->numa_ram_{b}"),
                        link_class=LinkClass.NUMA_INTERCONNECT,
                        source=f"numa_ram_{a}",
                        destination=f"numa_ram_{b}",
                        bidirectional=False,
                        peer_to_peer=True,
                        measured=False,
                    )
                )

        # One independent socket resource per discovered socket (never assume a single socket).
        for socket_idx in range(socket_count):
            affinity = tuple(
                f"numa_ram_{n}" for n in numa_nodes if socket_count == 1 or (n % socket_count) == socket_idx
            ) or tuple(f"numa_ram_{n}" for n in numa_nodes)
            graph.add_compute(
                ComputeResource(
                    id=ResourceId(ResourceKind.COMPUTE, f"cpu_socket_{socket_idx}"),
                    compute_class=ComputeClass.CPU_SOCKET,
                    backend_id=self.backend_id,
                    model=platform.processor() or platform.machine(),
                    architecture=platform.machine(),
                    vendor=platform.system(),
                    supported_dtypes=self._dtype_probe(),
                    supported_ops=self._op_probe(),
                    core_count=topo.cores_per_socket or max(1, physical // socket_count),
                    vector_isas=isas,
                    concurrency_limit=max(1, logical // socket_count),
                    numa_node=numa_nodes[min(socket_idx, len(numa_nodes) - 1)],
                    memory_affinity=affinity,
                    attributes={
                        "note": "per-socket view; prefer cpu_numa_* for placement",
                        "socket_index": socket_idx,
                    },
                )
            )
        return graph

    def _dtype_probe(self) -> tuple[str, ...]:
        dtypes = ["float32", "float64", "float16", "bfloat16", "int8", "int32", "int64", "bool"]
        return tuple(dtypes)

    def _op_probe(self) -> tuple[str, ...]:
        return (
            "aten::mm",
            "aten::addmm",
            "aten::bmm",
            "aten::convolution",
            "aten::relu",
            "aten::gelu",
            "aten::softmax",
            "aten::layer_norm",
            "aten::embedding",
            "aten::add",
            "aten::mul",
            "aten::cat",
            "aten::reshape",
        )

    def supported_ops(self, device: ComputeResource) -> tuple[str, ...]:
        return device.supported_ops or self._op_probe()

    def supported_dtypes(self, device: ComputeResource) -> tuple[str, ...]:
        return device.supported_dtypes or self._dtype_probe()

    def enumerate_kernels(
        self, region: RegionSource | Instruction | HeterogeneousGraph, device: ComputeResource
    ) -> list[KernelCandidate]:
        region_id = region_identifier(region)
        dtypes = [d for d in self.supported_dtypes(device) if d in ("float32",)]
        native = str(getattr(region, "attributes", {}).get("dtype", "float32"))
        if native not in dtypes:
            dtypes = [native]
        return [
            KernelCandidate(
                region_id=region_id,
                device=device.id.name,
                backend_id=self.backend_id,
                kernel_id=f"cpu_fx_{dtype}",
                dtype=dtype,
                attributes={"impl": "torch_fx_subgraph"},
            )
            for dtype in dtypes
        ]

    def benchmark(self, candidate: KernelCandidate) -> BenchmarkResult:
        """Measure a device-level matmul probe.

        Region-level measurement happens in :meth:`benchmark_region`; this probe
        only characterizes raw device throughput for planner priors.
        """
        import torch

        n = 256
        dtype = getattr(torch, candidate.dtype, torch.float32)
        a = torch.randn(n, n, dtype=dtype)
        b = torch.randn(n, n, dtype=dtype)
        for _ in range(3):
            torch.mm(a, b)
        start = time.perf_counter()
        iters = 10
        for _ in range(iters):
            torch.mm(a, b)
        elapsed = (time.perf_counter() - start) / iters
        return BenchmarkResult(
            candidate=candidate,
            latency_s=elapsed,
            memory_bytes=2 * n * n * a.element_size(),
            measured=True,
            notes=f"cpu matmul {n}x{n} {candidate.dtype}",
        )

    def benchmark_region(
        self,
        region: RegionSource,
        candidate: KernelCandidate,
        example_inputs: Sequence[Any],
        *,
        iters: int = 5,
    ) -> BenchmarkResult:
        compiled = self.compile(region, candidate)
        return benchmark_region_on_torch_device(candidate, compiled, list(example_inputs), iters=iters)

    def compile(self, region: RegionSource, candidate: KernelCandidate) -> CompiledRegion:
        return compile_region_for_torch_device(region, candidate, backend_id=self.backend_id, torch_device="cpu")

    def execute(self, executable: CompiledRegion, inputs: Sequence[Any]) -> tuple[Any, ...]:
        return execute_region_on_torch_device(executable, inputs)

    def transfer_capabilities(
        self, source: ComputeResource | str, destination: ComputeResource | str
    ) -> TransferCapability:
        src = source if isinstance(source, str) else source.id.name
        dst = destination if isinstance(destination, str) else destination.id.name
        return TransferCapability(
            source=src,
            destination=dst,
            kind="shared",
            notes="CPU memory moves use host shared-memory paths",
        )

    def resource_to_torch_device(self, resource_id: str) -> Any:
        import torch

        return torch.device("cpu")
