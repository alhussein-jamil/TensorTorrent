"""Measured decision about running independent regions concurrently.

Region-level parallelism only helps when a single region cannot saturate the
device. Rather than assuming it does, StreamCompiler times the widest independent
group of regions both ways and keeps the faster schedule. The measurement and the
resulting decision are recorded so nothing has to be taken on trust.
"""

from __future__ import annotations

import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Any

import torch

from streamcompiler.codegen.regions import Region, RegionProgram
from streamcompiler.parallel import inference_thread_pool


@dataclass
class ConcurrencyDecision:
    """Whether concurrent region execution was measured to help."""

    enabled: bool
    workers: int
    group: tuple[str, ...] = ()
    sequential_s: float = 0.0
    parallel_s: float = 0.0
    reason: str = ""
    measured: bool = False
    #: Intra-op threads each worker should use, or 0 to leave the process setting
    #: alone. Overlapping regions that each claim every core only contend.
    intraop_threads: int = 0

    @property
    def speedup(self) -> float:
        if self.parallel_s <= 0.0:
            return 0.0
        return self.sequential_s / self.parallel_s

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "workers": self.workers,
            "group": list(self.group),
            "sequential_s": self.sequential_s,
            "parallel_s": self.parallel_s,
            "speedup": self.speedup,
            "measured": self.measured,
            "intraop_threads": self.intraop_threads,
            "reason": self.reason,
        }


@dataclass
class DependencyLevels:
    """Regions grouped by dependency depth; one level is a set of independent regions."""

    levels: list[tuple[str, ...]] = field(default_factory=list)

    @property
    def width(self) -> int:
        return max((len(level) for level in self.levels), default=0)

    def widest(self) -> tuple[str, ...]:
        return max(self.levels, key=len, default=())


def dependency_levels(program: RegionProgram) -> DependencyLevels:
    """Compute dependency depth for every region.

    Regions sharing a level cannot depend on each other, directly or
    transitively, so they are always safe to run at the same time.
    """
    depth: dict[str, int] = {}
    for region in program.regions:
        depth[region.region_id] = max((depth[d] + 1 for d in region.depends_on if d in depth), default=0)
    grouped: dict[int, list[str]] = {}
    for region_id, level in depth.items():
        grouped.setdefault(level, []).append(region_id)
    return DependencyLevels(levels=[tuple(grouped[k]) for k in sorted(grouped)])


def transitive_dependencies(program: RegionProgram) -> dict[str, set[str]]:
    """Full ancestor set per region, used to validate observed overlaps."""
    ancestors: dict[str, set[str]] = {}
    for region in program.regions:
        acc: set[str] = set()
        for dep in region.depends_on:
            acc.add(dep)
            acc |= ancestors.get(dep, set())
        ancestors[region.region_id] = acc
    return ancestors


def measure_concurrency_benefit(
    program: RegionProgram,
    region_inputs: dict[str, tuple[Any, ...]],
    *,
    max_workers: int,
    iters: int = 3,
    min_speedup: float = 1.05,
) -> ConcurrencyDecision:
    """Time the widest independent region group sequentially and in parallel."""
    if max_workers <= 1:
        return ConcurrencyDecision(
            enabled=False,
            workers=1,
            reason="concurrency budget is one worker",
        )
    levels = dependency_levels(program)
    group_ids = levels.widest()
    if len(group_ids) < 2:
        return ConcurrencyDecision(
            enabled=False,
            workers=1,
            reason="graph has no independent regions to overlap",
        )
    group: list[Region] = []
    for region_id in group_ids:
        region = program.region_by_id(region_id)
        if region_inputs.get(region_id) is None or not region.node_count:
            continue
        group.append(region)
    if len(group) < 2:
        return ConcurrencyDecision(
            enabled=False,
            workers=1,
            group=group_ids,
            reason="independent regions have no measurable work",
        )

    calls = [(program.submodule(region), region_inputs[region.region_id]) for region in group]
    workers = min(max_workers, len(calls))
    process_threads = torch.get_num_threads()

    def run_sequential() -> None:
        for module, args in calls:
            module(*args)

    def run_parallel(pool: ThreadPoolExecutor) -> None:
        futures = [pool.submit(module, *args) for module, args in calls]
        for future in futures:
            future.result()

    best: tuple[float, int, int] | None = None
    attempts: list[str] = []
    with torch.inference_mode():
        run_sequential()
        sequential = min(_time(run_sequential) for _ in range(iters))
        try:
            for candidate_workers, threads in _candidates(workers, process_threads):
                torch.set_num_threads(threads)
                with inference_thread_pool(
                    max_workers=candidate_workers, thread_name_prefix="streamcompiler-measure"
                ) as pool:
                    run_parallel(pool)
                    elapsed = min(_time(lambda: run_parallel(pool)) for _ in range(iters))
                attempts.append(f"{candidate_workers}x{threads}t={elapsed * 1e3:.3f}ms")
                if best is None or elapsed < best[0]:
                    best = (elapsed, candidate_workers, threads)
        finally:
            torch.set_num_threads(process_threads)

    assert best is not None  # _candidates always yields at least one configuration
    parallel, chosen_workers, chosen_threads = best
    speedup = sequential / parallel if parallel > 0 else 0.0
    enabled = speedup >= min_speedup
    reason = (
        f"measured {speedup:.2f}x on {len(calls)} independent regions "
        f"({sequential * 1e3:.3f}ms sequential with {process_threads} threads vs "
        f"{parallel * 1e3:.3f}ms using {chosen_workers} workers x {chosen_threads} threads; "
        f"tried {', '.join(attempts)})"
    )
    if enabled:
        # A wide level can look faster while the full DAG pays more dispatch and
        # thread-pool tax than it gains. Confirm on the whole topo schedule.
        full_seq, full_par = _time_full_graph(
            program,
            region_inputs,
            workers=chosen_workers,
            threads=chosen_threads,
            iters=iters,
        )
        full_speedup = full_seq / full_par if full_par > 0 else 0.0
        reason += (
            f"; full-graph {full_speedup:.2f}x ({full_seq * 1e3:.3f}ms sequential vs {full_par * 1e3:.3f}ms parallel)"
        )
        if full_speedup < min_speedup:
            enabled = False
            reason += "; below the threshold on the full graph, so regions stay sequential"
    if not enabled and speedup < min_speedup:
        reason += "; below the threshold, so regions stay sequential"
    return ConcurrencyDecision(
        enabled=enabled,
        workers=chosen_workers if enabled else 1,
        group=tuple(r.region_id for r in group),
        sequential_s=sequential,
        parallel_s=parallel,
        reason=reason,
        measured=True,
        intraop_threads=chosen_threads if enabled else 0,
    )


def _time_full_graph(
    program: RegionProgram,
    region_inputs: dict[str, tuple[Any, ...]],
    *,
    workers: int,
    threads: int,
    iters: int,
) -> tuple[float, float]:
    """Wall-time the whole region DAG sequentially and with ``workers`` pools."""
    regions = [r for r in program.regions if region_inputs.get(r.region_id) is not None and r.node_count]
    if len(regions) < 2:
        return 0.0, 0.0
    calls = {r.region_id: (program.submodule(r), region_inputs[r.region_id]) for r in regions}
    dependents: dict[str, list[str]] = {r.region_id: [] for r in regions}
    pending = {r.region_id: set(r.depends_on) for r in regions}
    for region in regions:
        for dep in region.depends_on:
            if dep in dependents:
                dependents[dep].append(region.region_id)

    def run_sequential() -> None:
        for region in regions:
            module, args = calls[region.region_id]
            module(*args)

    def run_parallel(pool: ThreadPoolExecutor) -> None:
        left = {rid: set(deps) for rid, deps in pending.items()}
        ready = deque(rid for rid, deps in left.items() if not deps)
        running: dict[Any, str] = {}
        while ready or running:
            while ready and len(running) < workers:
                rid = ready.popleft()
                module, args = calls[rid]
                running[pool.submit(module, *args)] = rid
            if not running:
                continue
            done_set, _ = wait(list(running), return_when=FIRST_COMPLETED)
            for fut in done_set:
                rid = running.pop(fut)
                fut.result()
                for child in dependents.get(rid, ()):
                    left[child].discard(rid)
                    if not left[child]:
                        ready.append(child)

    process_threads = torch.get_num_threads()
    with torch.inference_mode():
        run_sequential()
        sequential = min(_time(run_sequential) for _ in range(iters))
        try:
            torch.set_num_threads(threads)
            with inference_thread_pool(max_workers=workers, thread_name_prefix="streamcompiler-full") as pool:
                run_parallel(pool)
                parallel = min(_time(lambda: run_parallel(pool)) for _ in range(iters))
        finally:
            torch.set_num_threads(process_threads)
    return sequential, parallel


def _candidates(workers: int, process_threads: int) -> list[tuple[int, int]]:
    """Worker/thread splits worth timing.

    Overlapping regions that each ask for every core mostly fight over the same
    cores, so the useful configurations divide the intra-op threads between the
    workers. The unsplit configuration is measured too, because it wins when the
    regions are small enough that thread startup dominates.
    """
    configs: list[tuple[int, int]] = []
    for candidate_workers in dict.fromkeys((workers, 2)):
        if candidate_workers < 2 or candidate_workers > workers:
            continue
        split = max(1, process_threads // candidate_workers)
        for threads in dict.fromkeys((split, process_threads)):
            configs.append((candidate_workers, threads))
    return configs or [(max(2, workers), process_threads)]


def _time(fn: Any) -> float:
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start
