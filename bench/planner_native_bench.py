#!/usr/bin/env python3
"""Repeatable native planner / batch-DES microbench.

Usage:
  uv run python bench/planner_native_bench.py
"""

from __future__ import annotations

import time

from tensortorrent.backends.base import KernelCandidate
from tensortorrent.config import CompileConfig
from tensortorrent.ir.graph import HeterogeneousGraph, Instruction, OpCode, TensorMeta
from tensortorrent.ir.resource_graph import (
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
from tensortorrent.planner.native import build_planning_problem, run_native_planner


def _machine(n_accel: int) -> ResourceGraph:
    machine = ResourceGraph(fingerprint=f"bench-{n_accel}", backends_present=("mock_accel", "cpu"))
    machine.add_memory(
        MemoryResource(
            id=ResourceId(ResourceKind.MEMORY, "host_ram"),
            memory_class=MemoryClass.NUMA_RAM,
            capacity_bytes=64 << 30,
            allocatable_bytes=64 << 30,
            attached_compute=("cpu0",),
        )
    )
    machine.add_compute(
        ComputeResource(
            id=ResourceId(ResourceKind.COMPUTE, "cpu0"),
            compute_class=ComputeClass.CPU_NUMA_POOL,
            backend_id="cpu",
            vendor="bench",
            model="cpu0",
            memory_affinity=("host_ram",),
            supported_dtypes=("float32",),
            core_count=8,
        )
    )
    for i in range(n_accel):
        mem = f"vram_{i}"
        dev = f"accel_{i}"
        machine.add_memory(
            MemoryResource(
                id=ResourceId(ResourceKind.MEMORY, mem),
                memory_class=MemoryClass.DEVICE_VRAM,
                capacity_bytes=8 << 30,
                allocatable_bytes=8 << 30,
                attached_compute=(dev,),
            )
        )
        machine.add_compute(
            ComputeResource(
                id=ResourceId(ResourceKind.COMPUTE, dev),
                compute_class=ComputeClass.ACCELERATOR,
                backend_id="mock_accel",
                vendor="bench",
                model=dev,
                memory_affinity=(mem,),
                supported_dtypes=("float32",),
            )
        )
        machine.add_link(
            TransferLink(
                id=ResourceId(ResourceKind.LINK, f"host_ram->{mem}"),
                link_class=LinkClass.PCIE,
                source="host_ram",
                destination=mem,
                bidirectional=True,
                measured=True,
                latency_s=8e-6,
                bytes_per_s=12e9,
            )
        )
    return machine


def _chain_graph(n_regions: int) -> HeterogeneousGraph:
    graph = HeterogeneousGraph(name=f"chain{n_regions}", outputs=(f"t{n_regions}",))
    graph.add_tensor(TensorMeta("t0", (1024,), "float32", size_bytes=4096, kind="input"))
    for i in range(n_regions):
        out = f"t{i + 1}"
        graph.add_tensor(TensorMeta(out, (1024,), "float32", size_bytes=4096, kind="activation"))
        deps = [f"r{i - 1}"] if i else []
        attrs = {"depends_on": deps} if deps else {}
        graph.add_instruction(Instruction(OpCode.COMPUTE, f"r{i}", inputs=(f"t{i}",), outputs=(out,), attributes=attrs))
    return graph


def _candidates(graph: HeterogeneousGraph, machine: ResourceGraph) -> dict[str, list[KernelCandidate]]:
    devices = [d for d in machine.compute.values() if d.compute_class != ComputeClass.COPY_ENGINE]
    out: dict[str, list[KernelCandidate]] = {}
    for region in graph.compute_regions():
        pool = []
        for device in devices:
            slow = 0.02 if device.compute_class == ComputeClass.CPU_NUMA_POOL else 0.002
            pool.append(
                KernelCandidate(
                    region_id=region.name,
                    device=device.id.name,
                    backend_id=device.backend_id,
                    kernel_id=f"{region.name}:{device.id.name}",
                    dtype="float32",
                    estimated_latency_s=slow,
                    attributes={"measured": True},
                )
            )
        out[region.name] = pool
    return out


def _byte_counts(graph: HeterogeneousGraph) -> dict[str, tuple[int, int]]:
    return {r.name: (4096, 0) for r in graph.compute_regions()}


def _subsets(machine: ResourceGraph):
    from tensortorrent.planner.maximal import _device_subsets

    eligible = [
        d
        for d in machine.compute.values()
        if d.compute_class in (ComputeClass.CPU_NUMA_POOL, ComputeClass.ACCELERATOR, ComputeClass.DISCRETE_GPU)
    ]
    return _device_subsets(eligible, limit=32)


def bench_case(name: str, n_accel: int, n_regions: int, workers: int, repeats: int = 5) -> dict:
    machine = _machine(n_accel)
    graph = _chain_graph(n_regions)
    cands = _candidates(graph, machine)
    bytes_ = _byte_counts(graph)
    subsets = _subsets(machine)
    cfg = CompileConfig(
        planner_beam_width=32,
        planner_candidates_per_device=2,
        planner_local_search_iters=1,
        planner_workers=workers,
        planner_parallel_subsets=True,
        planner_des_candidates=8,
        max_plan_candidates=32,
        measure_regions=False,
    )
    problem = build_planning_problem(graph, machine, cands, subsets, bytes_, cfg)
    assert problem is not None
    # warmup
    run_native_planner(problem)
    times = []
    stats = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = run_native_planner(problem)
        times.append(time.perf_counter() - t0)
        stats = out.get("statistics") or {}
    times.sort()
    return {
        "name": name,
        "workers": workers,
        "median_s": times[len(times) // 2],
        "min_s": times[0],
        "states_expanded": stats.get("states_expanded"),
        "parallel_search_used": stats.get("parallel_search_used"),
        "finalists": len(out.get("finalists") or []),
        "subsets": len(subsets),
        "regions": n_regions,
        "accels": n_accel,
    }


def main() -> None:
    cases = [
        ("1cpu_tiny", 0, 2, 1),
        ("1cpu_tiny_auto", 0, 2, 0),
        ("1accel_16r", 1, 16, 1),
        ("1accel_16r_auto", 1, 16, 0),
        ("2accel_16r", 2, 16, 1),
        ("2accel_16r_auto", 2, 16, 0),
        ("4accel_32r", 4, 32, 1),
        ("4accel_32r_auto", 4, 32, 0),
        ("8accel_64r", 8, 64, 1),
        ("8accel_64r_auto", 8, 64, 0),
    ]
    rows = [bench_case(*c) for c in cases]
    print(f"{'case':22} {'w':>3} {'median_ms':>10} {'expanded':>10} {'par':>5} {'final':>5}")
    for r in rows:
        print(
            f"{r['name']:22} {r['workers']:>3} {r['median_s'] * 1e3:10.3f} "
            f"{r['states_expanded']:>10} {str(r['parallel_search_used']):>5} {r['finalists']:>5}"
        )
    # Tiny auto overhead vs serial (absolute; relative % is noise below ~1ms).
    tiny_s = next(r for r in rows if r["name"] == "1cpu_tiny")
    tiny_a = next(r for r in rows if r["name"] == "1cpu_tiny_auto")
    delta_ms = (tiny_a["median_s"] - tiny_s["median_s"]) * 1e3
    print(f"tiny auto overhead vs serial: {delta_ms:+.3f} ms (both stay serial)")
    big_s = next(r for r in rows if r["name"] == "4accel_32r")
    big_a = next(r for r in rows if r["name"] == "4accel_32r_auto")
    if big_s["median_s"] > 0:
        speedup = big_s["median_s"] / max(big_a["median_s"], 1e-12)
        print(f"4accel_32r auto speedup vs serial: {speedup:.2f}x")


if __name__ == "__main__":
    main()
