"""Measured decision about running independent regions concurrently.

Region-level parallelism only helps when a single region cannot saturate the
device. Rather than assuming it does, StreamCompiler times the widest independent
group of regions both ways and keeps the faster schedule. The measurement and the
resulting decision are recorded so nothing has to be taken on trust.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
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

    def run_sequential() -> None:
        for module, args in calls:
            module(*args)

    def run_parallel(pool: ThreadPoolExecutor) -> None:
        futures = [pool.submit(module, *args) for module, args in calls]
        for future in futures:
            future.result()

    with torch.inference_mode():
        run_sequential()
        sequential = min(_time(run_sequential) for _ in range(iters))
        with inference_thread_pool(max_workers=workers, thread_name_prefix="streamcompiler-measure") as pool:
            run_parallel(pool)
            parallel = min(_time(lambda: run_parallel(pool)) for _ in range(iters))

    speedup = sequential / parallel if parallel > 0 else 0.0
    enabled = speedup >= min_speedup
    reason = (
        f"measured {speedup:.2f}x on {len(calls)} independent regions "
        f"({sequential * 1e3:.3f}ms sequential vs {parallel * 1e3:.3f}ms with {workers} workers)"
    )
    if not enabled:
        reason += "; below the threshold, so regions stay sequential"
    return ConcurrencyDecision(
        enabled=enabled,
        workers=workers if enabled else 1,
        group=tuple(r.region_id for r in group),
        sequential_s=sequential,
        parallel_s=parallel,
        reason=reason,
        measured=True,
    )


def _time(fn: Any) -> float:
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start
