"""Heterogeneous machine resource graph.

Every compute device, memory resource, and transfer path is an independent node
or edge. The planner must never collapse unequal devices into a homogeneous pool.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ResourceKind(str, Enum):
    COMPUTE = "compute"
    MEMORY = "memory"
    STORAGE = "storage"
    LINK = "link"


class ComputeClass(str, Enum):
    CPU_SOCKET = "cpu_socket"
    CPU_NUMA_POOL = "cpu_numa_pool"
    DISCRETE_GPU = "discrete_gpu"
    INTEGRATED_GPU = "integrated_gpu"
    ACCELERATOR = "accelerator"
    COPY_ENGINE = "copy_engine"


class MemoryClass(str, Enum):
    NUMA_RAM = "numa_ram"
    PINNED_HOST = "pinned_host"
    UNIFIED_SHARED = "unified_shared"
    DEVICE_VRAM = "device_vram"
    DISK_CACHE = "disk_cache"
    NVME = "nvme"


class LinkClass(str, Enum):
    CPU_LOCAL = "cpu_local"
    NUMA_INTERCONNECT = "numa_interconnect"
    PCIE = "pcie"
    NVLINK = "nvlink"
    INFINITY_FABRIC = "infinity_fabric"
    CXL = "cxl"
    SHARED_MEMORY = "shared_memory"
    HOST_STAGED = "host_staged"
    STORAGE = "storage"
    NETWORK = "network"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ResourceId:
    kind: ResourceKind
    name: str

    def __str__(self) -> str:
        return f"{self.kind.value}:{self.name}"


@dataclass
class ComputeResource:
    """Independent compute resource (socket, GPU, integrated GPU, copy engine, …)."""

    id: ResourceId
    compute_class: ComputeClass
    backend_id: str
    model: str
    architecture: str = ""
    vendor: str = ""
    # Capability fields are populated by backend discovery, never assumed.
    supported_dtypes: tuple[str, ...] = ()
    supported_ops: tuple[str, ...] = ()
    compute_capability: str = ""
    core_count: int = 0
    vector_isas: tuple[str, ...] = ()
    copy_engines: int = 0
    concurrency_limit: int = 1
    numa_node: int | None = None
    memory_affinity: tuple[str, ...] = ()  # MemoryResource names
    peak_flops_fp32: float | None = None  # informational only; never used as sole cost
    attributes: dict[str, Any] = field(default_factory=dict)

    def supports_dtype(self, dtype: str) -> bool:
        return dtype in self.supported_dtypes


@dataclass
class MemoryResource:
    """Independent memory or storage capacity."""

    id: ResourceId
    memory_class: MemoryClass
    capacity_bytes: int
    allocatable_bytes: int
    bandwidth_bytes_per_s: float | None = None  # theoretical; prefer measured links
    numa_node: int | None = None
    attached_compute: tuple[str, ...] = ()
    device_path: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class TransferLink:
    """Directed transfer path between two resources.

    Bandwidth and latency must come from measurement when available.
    Absence of a direct link does not make the machine unsupported —
    host-staged fallbacks are modeled explicitly.
    """

    id: ResourceId
    link_class: LinkClass
    source: str
    destination: str
    bidirectional: bool = False
    peer_to_peer: bool = False
    measured: bool = False
    # Size-dependent model coefficients when measured.
    latency_s: float | None = None
    bytes_per_s: float | None = None
    contention_factor: float = 1.0
    attributes: dict[str, Any] = field(default_factory=dict)

    def reverse_id(self) -> ResourceId:
        return ResourceId(ResourceKind.LINK, f"{self.destination}->{self.source}")


@dataclass
class ResourceDecision:
    """Planner explanation for including or excluding a resource."""

    resource: str
    selected: bool
    reason: str
    estimated_benefit_s: float | None = None
    estimated_cost_s: float | None = None


@dataclass
class ResourceGraph:
    """Full-machine heterogeneous resource graph discovered at runtime."""

    fingerprint: str
    compute: dict[str, ComputeResource] = field(default_factory=dict)
    memory: dict[str, MemoryResource] = field(default_factory=dict)
    links: dict[str, TransferLink] = field(default_factory=dict)
    backends_present: tuple[str, ...] = ()
    attributes: dict[str, Any] = field(default_factory=dict)

    def add_compute(self, node: ComputeResource) -> None:
        self.compute[node.id.name] = node

    def add_memory(self, node: MemoryResource) -> None:
        self.memory[node.id.name] = node

    def add_link(self, link: TransferLink) -> None:
        self.links[link.id.name] = link
        if link.bidirectional and link.reverse_id().name not in self.links:
            rev = TransferLink(
                id=link.reverse_id(),
                link_class=link.link_class,
                source=link.destination,
                destination=link.source,
                bidirectional=True,
                peer_to_peer=link.peer_to_peer,
                measured=link.measured,
                latency_s=link.latency_s,
                bytes_per_s=link.bytes_per_s,
                contention_factor=link.contention_factor,
                attributes=dict(link.attributes),
            )
            self.links[rev.id.name] = rev

    def compute_by_class(self, cls: ComputeClass) -> list[ComputeResource]:
        return [c for c in self.compute.values() if c.compute_class == cls]

    def gpus(self) -> list[ComputeResource]:
        return [
            c
            for c in self.compute.values()
            if c.compute_class in (ComputeClass.DISCRETE_GPU, ComputeClass.INTEGRATED_GPU)
        ]

    def cpu_sockets(self) -> list[ComputeResource]:
        return self.compute_by_class(ComputeClass.CPU_SOCKET)

    def memory_by_class(self, cls: MemoryClass) -> list[MemoryResource]:
        return [m for m in self.memory.values() if m.memory_class == cls]

    def link_between(self, source: str, destination: str) -> TransferLink | None:
        return self.links.get(f"{source}->{destination}")

    def has_direct_p2p(self, a: str, b: str) -> bool:
        link = self.link_between(a, b)
        return bool(link and link.peer_to_peer)

    def iter_resources(self) -> Iterator[str]:
        yield from self.compute
        yield from self.memory
        yield from self.links

    def summary(self) -> dict[str, Any]:
        return {
            "fingerprint": self.fingerprint,
            "backends": list(self.backends_present),
            "compute": {
                name: {
                    "class": c.compute_class.value,
                    "backend": c.backend_id,
                    "model": c.model,
                    "vendor": c.vendor,
                    "dtypes": list(c.supported_dtypes),
                    "numa": c.numa_node,
                }
                for name, c in self.compute.items()
            },
            "memory": {
                name: {
                    "class": m.memory_class.value,
                    "capacity_bytes": m.capacity_bytes,
                    "allocatable_bytes": m.allocatable_bytes,
                    "numa": m.numa_node,
                }
                for name, m in self.memory.items()
            },
            "links": {
                name: {
                    "class": link.link_class.value,
                    "source": link.source,
                    "destination": link.destination,
                    "p2p": link.peer_to_peer,
                    "measured": link.measured,
                    "bytes_per_s": link.bytes_per_s,
                }
                for name, link in self.links.items()
            },
        }

    def validate_independence(self) -> list[str]:
        """Return warnings if the graph incorrectly homogenizes unequal devices."""
        warnings: list[str] = []
        gpus = self.gpus()
        if len(gpus) >= 2:
            vendors = {g.vendor for g in gpus}
            mems = {
                next(
                    (self.memory[n].capacity_bytes for n in g.memory_affinity if n in self.memory),
                    None,
                )
                for g in gpus
            }
            if len(vendors) > 1 and "mixed_vendor" not in self.attributes:
                warnings.append("Multiple GPU vendors detected; planner must query each backend independently.")
            if len({m for m in mems if m is not None}) > 1:
                warnings.append("Unequal GPU memory capacities detected; shard sizing must be per-device.")
        return warnings


def merge_graphs(base: ResourceGraph, extra: ResourceGraph) -> ResourceGraph:
    """Merge backend-discovered subgraphs into one machine graph."""
    out = ResourceGraph(
        fingerprint=base.fingerprint or extra.fingerprint,
        backends_present=tuple(sorted(set(base.backends_present) | set(extra.backends_present))),
        attributes={**base.attributes, **extra.attributes},
    )
    for node in base.compute.values():
        out.add_compute(node)
    for node in extra.compute.values():
        out.add_compute(node)
    for node in base.memory.values():
        out.add_memory(node)
    for node in extra.memory.values():
        out.add_memory(node)
    for link in base.links.values():
        out.add_link(link)
    for link in extra.links.values():
        out.add_link(link)
    return out


def ensure_host_staged_fallbacks(graph: ResourceGraph) -> ResourceGraph:
    """Add explicit host-staged links where direct device-device paths are missing."""
    host_memories = [
        m
        for m in graph.memory.values()
        if m.memory_class in (MemoryClass.NUMA_RAM, MemoryClass.PINNED_HOST, MemoryClass.UNIFIED_SHARED)
    ]
    if not host_memories:
        return graph
    host = host_memories[0].id.name
    device_mems = [m for m in graph.memory.values() if m.memory_class == MemoryClass.DEVICE_VRAM]
    for a in device_mems:
        for b in device_mems:
            if a.id.name == b.id.name:
                continue
            direct = graph.link_between(a.id.name, b.id.name)
            if direct and direct.peer_to_peer:
                continue
            staged_name = f"{a.id.name}=host=>{b.id.name}"
            if staged_name in graph.links:
                continue
            graph.add_link(
                TransferLink(
                    id=ResourceId(ResourceKind.LINK, staged_name),
                    link_class=LinkClass.HOST_STAGED,
                    source=a.id.name,
                    destination=b.id.name,
                    bidirectional=False,
                    peer_to_peer=False,
                    measured=False,
                    attributes={"via": host, "fallback": True},
                )
            )
    return graph


def subset_names(resources: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted(resources))
