#!/usr/bin/env python3
"""Repeatable native planner / batch-DES microbench.

Usage:
  uv run python bench/planner_native_bench.py
"""

from __future__ import annotations

import time
from collections import Counter

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
from tensortorrent.runtime.schedule import ExecutableSchedule, PlanInstruction
from tensortorrent.runtime.simulator.discrete_event import (
    simulate_schedule,
    simulate_schedules,
    simulate_schedules_with_stats,
)


def _machine(n_accel: int, *, vram: int = 8 << 30) -> ResourceGraph:
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
                capacity_bytes=vram,
                allocatable_bytes=vram,
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
        if i > 0:
            machine.add_link(
                TransferLink(
                    id=ResourceId(ResourceKind.LINK, f"vram_0->vram_{i}"),
                    link_class=LinkClass.PCIE,
                    source="vram_0",
                    destination=mem,
                    bidirectional=True,
                    measured=True,
                    latency_s=1e-5,
                    bytes_per_s=8e9,
                    contention_factor=1.5,
                )
            )
    return machine


def _chain_graph(n_regions: int, *, bytes_per: int = 4096) -> HeterogeneousGraph:
    graph = HeterogeneousGraph(name=f"chain{n_regions}", outputs=(f"t{n_regions}",))
    graph.add_tensor(TensorMeta("t0", (bytes_per // 4,), "float32", size_bytes=bytes_per, kind="input"))
    for i in range(n_regions):
        out = f"t{i + 1}"
        graph.add_tensor(TensorMeta(out, (bytes_per // 4,), "float32", size_bytes=bytes_per, kind="activation"))
        deps = [f"r{i - 1}"] if i else []
        attrs = {"depends_on": deps} if deps else {}
        graph.add_instruction(Instruction(OpCode.COMPUTE, f"r{i}", inputs=(f"t{i}",), outputs=(out,), attributes=attrs))
    return graph


def _fork_join_graph(n_branches: int = 4) -> HeterogeneousGraph:
    graph = HeterogeneousGraph(name=f"fork{n_branches}", outputs=("y",))
    graph.add_tensor(TensorMeta("x", (256,), "float32", size_bytes=1024, kind="input"))
    graph.add_instruction(Instruction(OpCode.COMPUTE, "root", inputs=("x",), outputs=("m0",)))
    graph.add_tensor(TensorMeta("m0", (256,), "float32", size_bytes=1024, kind="activation"))
    mids = []
    for i in range(n_branches):
        mid = f"b{i}"
        mids.append(mid)
        graph.add_tensor(TensorMeta(mid, (256,), "float32", size_bytes=1024, kind="activation"))
        graph.add_instruction(
            Instruction(
                OpCode.COMPUTE,
                f"br{i}",
                inputs=("m0",),
                outputs=(mid,),
                attributes={"depends_on": ["root"]},
            )
        )
    graph.add_tensor(TensorMeta("y", (256,), "float32", size_bytes=1024, kind="activation"))
    graph.add_instruction(
        Instruction(
            OpCode.COMPUTE,
            "join",
            inputs=tuple(mids),
            outputs=("y",),
            attributes={"depends_on": [f"br{i}" for i in range(n_branches)]},
        )
    )
    return graph


def _candidates(
    graph: HeterogeneousGraph,
    machine: ResourceGraph,
    *,
    per_device: int = 2,
) -> dict[str, list[KernelCandidate]]:
    devices = [d for d in machine.compute.values() if d.compute_class != ComputeClass.COPY_ENGINE]
    out: dict[str, list[KernelCandidate]] = {}
    for region in graph.compute_regions():
        pool = []
        for device in devices:
            for k in range(per_device):
                slow = 0.02 if device.compute_class == ComputeClass.CPU_NUMA_POOL else 0.002
                pool.append(
                    KernelCandidate(
                        region_id=region.name,
                        device=device.id.name,
                        backend_id=device.backend_id,
                        kernel_id=f"{region.name}:{device.id.name}:k{k}",
                        dtype="float32",
                        estimated_latency_s=slow * (1.0 + 0.1 * k),
                        workspace_bytes=0 if k == 0 else 256,
                        attributes={"measured": True},
                    )
                )
        out[region.name] = pool
    return out


def _byte_counts(graph: HeterogeneousGraph, nbytes: int = 4096) -> dict[str, tuple[int, int]]:
    return {r.name: (nbytes, 0) for r in graph.compute_regions()}


def _subsets(machine: ResourceGraph):
    from tensortorrent.planner.maximal import _device_subsets

    eligible = [
        d
        for d in machine.compute.values()
        if d.compute_class in (ComputeClass.CPU_NUMA_POOL, ComputeClass.ACCELERATOR, ComputeClass.DISCRETE_GPU)
    ]
    return _device_subsets(eligible, limit=32)


def bench_case(
    name: str,
    n_accel: int,
    n_regions: int,
    workers: int,
    *,
    repeats: int = 5,
    graph_kind: str = "chain",
    vram: int = 8 << 30,
    beam: int = 32,
    cands: int = 2,
) -> dict:
    machine = _machine(n_accel, vram=vram)
    graph = _fork_join_graph(max(2, n_regions // 2)) if graph_kind == "fork" else _chain_graph(n_regions)
    cand_map = _candidates(graph, machine, per_device=cands)
    bytes_ = _byte_counts(graph)
    subsets = _subsets(machine)
    cfg = CompileConfig(
        planner_beam_width=beam,
        planner_candidates_per_device=cands,
        planner_local_search_iters=1,
        planner_workers=workers,
        planner_parallel_subsets=True,
        planner_des_candidates=8,
        planner_per_subset_finalists=4,
        max_plan_candidates=32,
        measure_regions=False,
    )
    t_conv0 = time.perf_counter()
    problem = build_planning_problem(graph, machine, cand_map, subsets, bytes_, cfg)
    conv_s = time.perf_counter() - t_conv0
    assert problem is not None
    run_native_planner(problem)
    times = []
    stats = None
    out = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = run_native_planner(problem)
        times.append(time.perf_counter() - t0)
        stats = out.get("statistics") or {}
    times.sort()
    finalists = list((out or {}).get("finalists") or [])
    subset_hist = Counter(tuple(f.get("subset_devices") or ()) for f in finalists)
    max_same_subset = max(subset_hist.values(), default=0)

    # Batch DES microbench on synthetic schedules.
    scheds = []
    for i in range(min(8, max(1, len(finalists)))):
        scheds.append(
            ExecutableSchedule(
                graph_name="b",
                fingerprint="fp",
                instructions=(
                    PlanInstruction(
                        opcode=OpCode.COMPUTE,
                        name=f"c{i}",
                        resource="cpu0" if n_accel == 0 else "accel_0",
                        outputs=(f"o{i}",),
                        nbytes=64,
                        predicted_duration_s=0.01 * (i + 1),
                    ),
                ),
            )
        )
    if not scheds:
        scheds = [
            ExecutableSchedule(
                graph_name="b",
                fingerprint="fp",
                instructions=(
                    PlanInstruction(
                        opcode=OpCode.COMPUTE,
                        name="c0",
                        resource="cpu0",
                        outputs=("o0",),
                        nbytes=64,
                        predicted_duration_s=0.01,
                    ),
                ),
            )
        ]
    t0 = time.perf_counter()
    _ = simulate_schedule(scheds[0], machine)
    scalar_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    _ = simulate_schedules(scheds, machine, workers=1)
    batch_serial_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    _, auto_stats = simulate_schedules_with_stats(scheds, machine, workers=0)
    batch_auto_s = time.perf_counter() - t0

    return {
        "name": name,
        "workers": workers,
        "median_s": times[len(times) // 2],
        "min_s": times[0],
        "conv_s": conv_s,
        "states_expanded": stats.get("states_expanded"),
        "parallel_search_used": stats.get("parallel_search_used"),
        "parallel_beam_used": stats.get("parallel_beam_used"),
        "planner_workers_requested": stats.get("planner_workers_requested"),
        "planner_workers_available": stats.get("planner_workers_available"),
        "planner_workers_used": stats.get("planner_workers_used"),
        "planner_pool_threads": stats.get("planner_pool_threads"),
        "finalists": len(finalists),
        "max_same_subset_finalists": max_same_subset,
        "subsets": len(subsets),
        "regions": len(list(graph.compute_regions())),
        "accels": n_accel,
        "scalar_des_s": scalar_s,
        "batch_des_serial_s": batch_serial_s,
        "batch_des_auto_s": batch_auto_s,
        "des_variants": len(scheds),
        "parallel_simulation_used": auto_stats.get("parallel_simulation_used"),
        "simulator_workers_used": auto_stats.get("simulator_workers_used"),
    }


def main() -> None:
    cases = [
        ("1cpu_tiny", 0, 2, 1, {}),
        ("1cpu_tiny_auto", 0, 2, 0, {}),
        ("1accel_4r", 1, 4, 1, {}),
        ("1accel_16r", 1, 16, 1, {}),
        ("1accel_16r_auto", 1, 16, 0, {}),
        ("1accel_16r_c4", 1, 16, 1, {"cands": 4}),
        ("2accel_16r", 2, 16, 1, {}),
        ("2accel_16r_auto", 2, 16, 0, {}),
        ("2accel_fork", 2, 8, 1, {"graph_kind": "fork"}),
        ("4accel_32r", 4, 32, 1, {}),
        ("4accel_32r_auto", 4, 32, 0, {}),
        ("4accel_32r_mem", 4, 32, 1, {"vram": 1 << 20}),
        ("4accel_64r_beam128", 4, 64, 1, {"beam": 128}),
        ("8accel_64r", 8, 64, 1, {}),
        ("8accel_64r_auto", 8, 64, 0, {}),
        ("8accel_128r", 8, 128, 1, {}),
    ]
    rows = []
    for name, n_accel, n_regions, workers, kw in cases:
        rows.append(bench_case(name, n_accel, n_regions, workers, **kw))
    print(
        f"{'case':22} {'w':>3} {'med_ms':>8} {'conv_ms':>8} {'exp':>9} "
        f"{'par':>5} {'beam':>5} {'fin':>4} {'same':>4} {'bDES_ms':>8}"
    )
    for r in rows:
        print(
            f"{r['name']:22} {r['workers']:>3} {r['median_s'] * 1e3:8.3f} "
            f"{r['conv_s'] * 1e3:8.3f} {r['states_expanded'] or 0:>9} "
            f"{str(r['parallel_search_used']):>5} {str(r['parallel_beam_used']):>5} "
            f"{r['finalists']:>4} {r['max_same_subset_finalists']:>4} "
            f"{r['batch_des_auto_s'] * 1e3:8.3f}"
        )
    tiny_s = next(r for r in rows if r["name"] == "1cpu_tiny")
    tiny_a = next(r for r in rows if r["name"] == "1cpu_tiny_auto")
    print(f"tiny auto overhead: {(tiny_a['median_s'] - tiny_s['median_s']) * 1e3:+.3f} ms")
    big_s = next(r for r in rows if r["name"] == "4accel_32r")
    big_a = next(r for r in rows if r["name"] == "4accel_32r_auto")
    print(f"4accel_32r auto speedup: {big_s['median_s'] / max(big_a['median_s'], 1e-12):.2f}x")
    print(
        f"DES scalar={tiny_s['scalar_des_s'] * 1e3:.3f}ms "
        f"batch_serial={tiny_s['batch_des_serial_s'] * 1e3:.3f}ms "
        f"batch_auto={tiny_s['batch_des_auto_s'] * 1e3:.3f}ms "
        f"n={tiny_s['des_variants']}"
    )


if __name__ == "__main__":
    main()
