"""Shared test helpers.

CPU/mock tests must not accidentally place on a real GPU when one is present.
NVIDIA-specific coverage lives under ``tests/hardware/`` with the ``gpu`` marker.
"""

from __future__ import annotations

from tensortorrent.config import CompileConfig
from tensortorrent.hardware.discovery import discover_resource_graph
from tensortorrent.ir.resource_graph import (
    ComputeClass,
    MemoryClass,
    ResourceGraph,
    ResourceId,
    ResourceKind,
    TransferLink,
)


def cpu_host_graph() -> ResourceGraph:
    """Discovery graph with discrete/integrated GPUs and their VRAM stripped."""
    base = discover_resource_graph()
    gpu_names = {
        name
        for name, node in base.compute.items()
        if node.compute_class in (ComputeClass.DISCRETE_GPU, ComputeClass.INTEGRATED_GPU)
    }
    vram_names = {
        name
        for name, mem in base.memory.items()
        if mem.memory_class == MemoryClass.DEVICE_VRAM or any(c in gpu_names for c in mem.attached_compute)
    }
    drop = gpu_names | vram_names
    out = ResourceGraph(
        fingerprint=f"{base.fingerprint}-cpu-host",
        backends_present=tuple(b for b in base.backends_present if b not in ("cuda", "rocm")),
        attributes=dict(base.attributes),
    )
    for name, node in base.compute.items():
        if name in drop:
            continue
        out.add_compute(node)
    for name, mem in base.memory.items():
        if name in drop:
            continue
        out.add_memory(mem)
    for name, link in base.links.items():
        if link.source in drop or link.destination in drop:
            continue
        out.add_link(
            TransferLink(
                id=ResourceId(ResourceKind.LINK, name),
                link_class=link.link_class,
                source=link.source,
                destination=link.destination,
                bidirectional=link.bidirectional,
                peer_to_peer=link.peer_to_peer,
                measured=link.measured,
                latency_s=link.latency_s,
                bytes_per_s=link.bytes_per_s,
                contention_factor=link.contention_factor,
                attributes=dict(link.attributes),
            )
        )
    return out


def cpu_config(**kwargs) -> CompileConfig:
    """CompileConfig that refuses discrete/integrated GPUs."""
    kwargs.setdefault("allow_gpu", False)
    kwargs.setdefault("allow_integrated_gpu", False)
    return CompileConfig(**kwargs)
