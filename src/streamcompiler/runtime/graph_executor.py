"""Dependency-aware region executor.

This is the component that actually runs a model. It owns a tensor environment
keyed by the value names produced during lowering, dispatches each region to the
backend that the planner selected, and releases activations as soon as their last
consumer finishes.

Concurrency rules
-----------------
* A region starts only after every region it depends on has completed.
* Regions with no outstanding dependencies may run on different workers at the
  same time. PyTorch releases the GIL inside its kernels, so independent CPU
  regions genuinely overlap.
* Parallelism is limited to the number of distinct devices the plan selected
  (never more), so single-device plans keep their sequential fast path.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Any

import torch

from streamcompiler.codegen.regions import Region, RegionBinding, RegionProgram
from streamcompiler.errors import RuntimePlanError
from streamcompiler.runtime.tensor_store import ParameterStore


@dataclass
class RegionEvent:
    region_id: str
    device: str
    backend_id: str
    start_s: float
    end_s: float
    worker: str

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass
class ExecutionReport:
    wall_time_s: float
    events: list[RegionEvent] = field(default_factory=list)
    peak_activation_bytes: int = 0
    released_values: int = 0
    parallel_overlaps: int = 0
    max_concurrent_regions: int = 1
    parameter_store: dict[str, Any] = field(default_factory=dict)

    def overlapping_pairs(self) -> list[tuple[str, str]]:
        """Region pairs whose execution intervals genuinely overlapped in time."""
        pairs: list[tuple[str, str]] = []
        ordered = sorted(self.events, key=lambda e: e.start_s)
        for i, first in enumerate(ordered):
            for second in ordered[i + 1 :]:
                if second.start_s >= first.end_s:
                    break
                pairs.append((first.region_id, second.region_id))
        return pairs

    def as_dict(self) -> dict[str, Any]:
        return {
            "wall_time_s": self.wall_time_s,
            "region_count": len(self.events),
            "peak_activation_bytes": self.peak_activation_bytes,
            "released_values": self.released_values,
            "parallel_overlaps": self.parallel_overlaps,
            "max_concurrent_regions": self.max_concurrent_regions,
            "parameter_store": self.parameter_store,
            "regions": [
                {
                    "region_id": e.region_id,
                    "device": e.device,
                    "backend_id": e.backend_id,
                    "duration_s": e.duration_s,
                    "worker": e.worker,
                }
                for e in self.events
            ],
        }


class GraphExecutor:
    """Executes a :class:`RegionProgram` using pre-compiled region bindings."""

    def __init__(
        self,
        program: RegionProgram,
        bindings: dict[str, RegionBinding],
        *,
        parameter_store: ParameterStore,
        max_workers: int = 1,
        prefetch_distance: int = 1,
    ) -> None:
        missing = [r.region_id for r in program.regions if r.region_id not in bindings]
        if missing:
            raise RuntimePlanError(f"No compiled executable for regions: {missing}")
        self.program = program
        self.bindings = bindings
        self.parameter_store = parameter_store
        self.max_workers = max(1, int(max_workers))
        self.prefetch_distance = max(0, int(prefetch_distance))
        self._consumers = self._count_consumers()
        self._dependents = self._build_dependents()
        self._lock = threading.Lock()

    # ---- static graph facts ----------------------------------------
    def _count_consumers(self) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for region in self.program.regions:
            for name in region.inputs:
                counts[name] += 1
        for kind, ref in self.program.output_refs:
            if kind == "value":
                counts[str(ref)] += 1
        return dict(counts)

    def _build_dependents(self) -> dict[str, list[str]]:
        dependents: dict[str, list[str]] = defaultdict(list)
        for region in self.program.regions:
            for dep in region.depends_on:
                dependents[dep].append(region.region_id)
        return dict(dependents)

    # ---- execution --------------------------------------------------
    def run(self, flat_inputs: list[Any]) -> tuple[list[Any], ExecutionReport]:
        program = self.program
        env: dict[str, Any] = {}
        remaining = dict(self._consumers)
        report = ExecutionReport(wall_time_s=0.0)
        activation_bytes = 0
        peak_bytes = 0
        released = 0
        state_ready: set[str] = set()

        for name, value in zip(program.user_inputs, flat_inputs, strict=True):
            env[name] = value

        pending_deps = {r.region_id: set(r.depends_on) for r in program.regions}
        order = {r.region_id: i for i, r in enumerate(program.regions)}
        ready: deque[Region] = deque(r for r in program.regions if not pending_deps[r.region_id])
        if not ready and program.regions:
            raise RuntimePlanError("Region dependency graph has no entry point (cycle detected)")

        self._prefetch_from(0)
        start_wall = time.perf_counter()

        executor: ThreadPoolExecutor | None = None
        if self.max_workers > 1 and len(program.regions) > 1:
            executor = ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="streamcompiler-region")
        running: dict[Future[tuple[RegionEvent, tuple[Any, ...]]], Region] = {}

        def complete(region: Region, event: RegionEvent, outputs: tuple[Any, ...]) -> None:
            nonlocal activation_bytes, peak_bytes, released
            report.events.append(event)
            if not region.outputs:
                # Side-effect-only region (shape guards); it binds no values.
                outputs = ()
            if len(outputs) != len(region.outputs):
                raise RuntimePlanError(
                    f"Region {region.region_id} produced {len(outputs)} values, plan expects {len(region.outputs)}"
                )
            for name, value in zip(region.outputs, outputs, strict=True):
                env[name] = value
                if isinstance(value, torch.Tensor):
                    activation_bytes += value.numel() * value.element_size()
            peak_bytes = max(peak_bytes, activation_bytes)
            freed, count = self._release_inputs(region, env, remaining, state_ready)
            activation_bytes -= freed
            released += count
            for dependent in self._dependents.get(region.region_id, []):
                pending_deps[dependent].discard(region.region_id)
                if not pending_deps[dependent]:
                    ready.append(program.region_by_id(dependent))
            nxt = min((order[d] for d in self._dependents.get(region.region_id, [])), default=None)
            if nxt is not None:
                self._prefetch_from(nxt)

        try:
            while ready or running:
                while ready and (executor is None or len(running) < self.max_workers):
                    region = ready.popleft()
                    args = self._gather_inputs(region, env)
                    if executor is None:
                        event, outputs = self._run_region(region, args)
                        complete(region, event, outputs)
                    else:
                        running[executor.submit(self._run_region, region, args)] = region
                if not running:
                    continue
                report.max_concurrent_regions = max(report.max_concurrent_regions, len(running))
                done, _ = wait(list(running), return_when=FIRST_COMPLETED)
                for future in done:
                    finished = running.pop(future)
                    event, outputs = future.result()
                    complete(finished, event, outputs)
        finally:
            if executor is not None:
                executor.shutdown(wait=True)

        report.wall_time_s = time.perf_counter() - start_wall
        report.peak_activation_bytes = peak_bytes
        report.released_values = released
        report.parallel_overlaps = len(report.overlapping_pairs())
        report.parameter_store = self.parameter_store.stats()

        flat_outputs: list[Any] = []
        for kind, ref in program.output_refs:
            if kind == "constant":
                flat_outputs.append(ref)
                continue
            name = str(ref)
            if name not in env:
                raise RuntimePlanError(f"Output value {name} was never produced")
            flat_outputs.append(env[name])
        return flat_outputs, report

    # ---- helpers ----------------------------------------------------
    def _gather_inputs(self, region: Region, env: dict[str, Any]) -> tuple[Any, ...]:
        args: list[Any] = []
        for name in region.inputs:
            if name in env:
                args.append(env[name])
            elif name in self.program.state_bindings:
                tensor = self.parameter_store.acquire(name)
                env[name] = tensor
                args.append(tensor)
            else:
                raise RuntimePlanError(
                    f"Region {region.region_id} needs value {name} which is not available; "
                    "the execution plan has an invalid dependency order"
                )
        return tuple(args)

    def _run_region(self, region: Region, args: tuple[Any, ...]) -> tuple[RegionEvent, tuple[Any, ...]]:
        binding = self.bindings[region.region_id]
        from streamcompiler.backends import backend_by_id

        backend = backend_by_id(binding.backend_id)
        if backend is None:
            raise RuntimePlanError(f"Backend {binding.backend_id} is not registered")
        start = time.perf_counter()
        with torch.inference_mode():
            outputs = backend.execute(binding.compiled, args)
        end = time.perf_counter()
        event = RegionEvent(
            region_id=region.region_id,
            device=binding.device,
            backend_id=binding.backend_id,
            start_s=start,
            end_s=end,
            worker=threading.current_thread().name,
        )
        return event, outputs

    def _release_inputs(
        self,
        region: Region,
        env: dict[str, Any],
        remaining: dict[str, int],
        state_ready: set[str],
    ) -> tuple[int, int]:
        freed = 0
        count = 0
        finished_state: list[str] = []
        for name in region.inputs:
            if name not in remaining:
                continue
            remaining[name] -= 1
            if remaining[name] > 0:
                continue
            value = env.pop(name, None)
            if name in self.program.state_bindings:
                finished_state.append(name)
            elif isinstance(value, torch.Tensor):
                freed += value.numel() * value.element_size()
            count += 1
        if finished_state:
            self.parameter_store.release(tuple(finished_state))
            state_ready.difference_update(finished_state)
        return freed, count

    def _prefetch_from(self, index: int) -> None:
        if self.prefetch_distance == 0:
            return
        regions = self.program.regions
        names: list[str] = []
        for region in regions[index : index + self.prefetch_distance + 1]:
            names.extend(region.state_inputs)
        if names:
            self.parameter_store.prefetch(tuple(dict.fromkeys(names)))
