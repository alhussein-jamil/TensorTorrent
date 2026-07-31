"""Whole-machine hardware discovery via backend contracts and host topology."""

from __future__ import annotations

import os
from pathlib import Path

import psutil

from streamcompiler.backends import all_backends
from streamcompiler.hardware.fingerprint import collect_fingerprint_payload, machine_fingerprint
from streamcompiler.ir.resource_graph import (
    LinkClass,
    MemoryClass,
    MemoryResource,
    ResourceGraph,
    ResourceId,
    ResourceKind,
    TransferLink,
    ensure_host_staged_fallbacks,
    merge_graphs,
)


def _discover_storage(graph: ResourceGraph) -> None:
    """Represent each distinct mount/device as an independent storage resource."""
    seen: set[str] = set()
    index = 0
    for part in psutil.disk_partitions(all=False):
        if part.fstype in ("", "tmpfs", "devtmpfs", "squashfs", "overlay"):
            continue
        key = part.device or part.mountpoint
        if key in seen:
            continue
        seen.add(key)
        try:
            usage = psutil.disk_usage(part.mountpoint)
        except OSError:
            continue
        name = f"storage_{index}"
        # Prefer NVMe naming when the device path suggests it.
        is_nvme = "nvme" in (part.device or "").lower() or "nvme" in part.mountpoint.lower()
        mem_class = MemoryClass.NVME if is_nvme else MemoryClass.DISK_CACHE
        graph.add_memory(
            MemoryResource(
                id=ResourceId(ResourceKind.MEMORY, name),
                memory_class=mem_class,
                capacity_bytes=int(usage.total),
                allocatable_bytes=int(usage.free),
                device_path=part.device,
                attributes={"mountpoint": part.mountpoint, "fstype": part.fstype},
            )
        )
        # Link from primary host memory to this storage device.
        host = next(iter(graph.memory_by_class(MemoryClass.NUMA_RAM)), None)
        if host is not None:
            graph.add_link(
                TransferLink(
                    id=ResourceId(ResourceKind.LINK, f"{host.id.name}->{name}"),
                    link_class=LinkClass.STORAGE,
                    source=host.id.name,
                    destination=name,
                    bidirectional=True,
                    peer_to_peer=False,
                    measured=False,
                )
            )
        index += 1


def discover_resource_graph() -> ResourceGraph:
    """Discover the actual machine as a heterogeneous resource graph.

    This must be called on the deployment machine. Results from a CPU-only
    development host must not be treated as production GPU validation.
    """
    payload = collect_fingerprint_payload()
    fp = machine_fingerprint(payload)
    graph = ResourceGraph(
        fingerprint=fp,
        attributes={
            "discovery_host": os.uname().nodename if hasattr(os, "uname") else "",
            "fingerprint_payload_keys": sorted(payload.keys()),
            "dev_machine_note": ("Absence of accelerators here does not prove accelerator paths work elsewhere."),
        },
    )

    present: list[str] = []
    for backend in all_backends():
        try:
            available = backend.available()
        except Exception as exc:  # noqa: BLE001
            graph.attributes[f"{backend.backend_id}_available_error"] = str(exc)
            continue
        graph.attributes[f"{backend.backend_id}_available"] = available
        if not available and backend.backend_id != "cpu":
            continue
        try:
            sub = backend.discover_devices()
        except Exception as exc:  # noqa: BLE001
            graph.attributes[f"{backend.backend_id}_discover_error"] = str(exc)
            continue
        if (
            backend.backend_id not in sub.backends_present
            and available
            and (sub.compute or backend.backend_id == "cpu")
        ):
            # CPU always contributes; others only when devices found.
            sub.backends_present = tuple(sorted(set(sub.backends_present) | {backend.backend_id}))
        graph = merge_graphs(graph, sub)
        present.extend(sub.backends_present)

    graph.backends_present = tuple(sorted(set(present)))
    graph.fingerprint = fp
    for device in graph.compute.values():
        device.attributes.setdefault("machine_fingerprint", fp)
        device.attributes.setdefault("fingerprint", f"{device.id.name}:{fp[:12]}")
    _discover_storage(graph)
    graph = ensure_host_staged_fallbacks(graph)
    graph.attributes["independence_warnings"] = graph.validate_independence()
    return graph


def write_discovery_report(graph: ResourceGraph, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    import json

    path.write_text(json.dumps(graph.summary(), indent=2, sort_keys=True), encoding="utf-8")
