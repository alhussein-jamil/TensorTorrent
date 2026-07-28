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

from streamcompiler.backends.torch_device import coerce_region_result, unwrap_region_callable
from streamcompiler.codegen.regions import Region, RegionBinding, RegionProgram
from streamcompiler.errors import RuntimePlanError
from streamcompiler.parallel import inference_thread_pool
from streamcompiler.runtime.schedule import ExecutableSchedule, MemoryTier
from streamcompiler.runtime.tensor_directory import TensorDirectory
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
class _FastPath:
    """Pre-bound single-region call: user inputs swap in, state stays put."""

    region_id: str
    device: str
    backend_id: str
    call: Any
    args: list[Any]
    user_slots: tuple[tuple[int, int], ...]
    state_bytes: int


@dataclass
class _StaticResident:
    """Pre-bound multi-region sequential walk with resident state pinned once."""

    steps: tuple[tuple[Region, Any, tuple[str, ...]], ...]
    activation_names: tuple[str, ...]
    env: dict[str, Any]


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
        intraop_threads: int = 0,
        activation_budget_bytes: int | None = None,
        schedule: ExecutableSchedule | None = None,
        tensor_directory: TensorDirectory | None = None,
    ) -> None:
        missing = [r.region_id for r in program.regions if r.region_id not in bindings]
        if missing:
            raise RuntimePlanError(f"No compiled executable for regions: {missing}")
        self.program = program
        self.bindings = bindings
        self.parameter_store = parameter_store
        self.max_workers = max(1, int(max_workers))
        self.prefetch_distance = max(0, int(prefetch_distance))
        self.intraop_threads = max(0, int(intraop_threads))
        self.activation_budget_bytes = activation_budget_bytes
        self.schedule = schedule
        self.tensor_directory = tensor_directory if tensor_directory is not None else TensorDirectory()
        if schedule is not None:
            scheduled = [i.executable_ref for i in schedule.compute_ops() if i.executable_ref]
            actual = [r.region_id for r in program.regions]
            if scheduled != actual:
                raise RuntimePlanError(
                    f"ExecutableSchedule compute order {scheduled} does not match program regions {actual}"
                )
        self._consumers = self._count_consumers()
        self._dependents = self._build_dependents()
        self._order = {r.region_id: i for i, r in enumerate(program.regions)}
        self._static_order = self._verified_static_order()
        self._prefetch_enabled = self.prefetch_distance > 0 and parameter_store.needs_prefetch
        self._callables = self._resolve_callables()
        self._fast = self._build_fast_path()
        self._static_resident = self._build_static_resident() if self._fast is None else None
        self._run_lock = threading.Lock()
        self._transfer_events: list[dict[str, Any]] = []

    def _check_activation_budget(self, peak_bytes: int) -> None:
        budget = self.activation_budget_bytes
        if budget is not None and peak_bytes > budget:
            raise RuntimePlanError(f"Peak live activations {peak_bytes} bytes exceed activation_budget_bytes={budget}")

    @property
    def uses_fast_path(self) -> bool:
        """True when calls skip the general scheduler for a single resident region."""
        return self._fast is not None

    @property
    def uses_static_resident(self) -> bool:
        """True when calls walk a pre-bound multi-region resident plan."""
        return self._static_resident is not None

    # ---- static graph facts ----------------------------------------
    def _build_static_resident(self) -> _StaticResident | None:
        """Prebind state for multi-region single-worker resident plans."""
        if self.max_workers != 1 or self._static_order is None or self.parameter_store.needs_prefetch:
            return None
        if len(self.program.regions) < 2:
            return None
        resident_state = {name: self.parameter_store.acquire(name) for name in self.program.state_bindings}
        activation_names = tuple(
            name
            for region in self._static_order
            for name in region.outputs
            if name not in resident_state and name not in self.program.user_inputs
        )
        steps = tuple((region, self._callables[region.region_id], region.inputs) for region in self._static_order)
        return _StaticResident(steps=steps, activation_names=activation_names, env=dict(resident_state))

    def _resolve_callables(self) -> dict[str, Any]:
        """Cache the callable executable for each region once."""
        out: dict[str, Any] = {}
        for region_id, binding in self.bindings.items():
            compiled = binding.compiled
            executable = getattr(compiled, "executable", compiled)
            executable = unwrap_region_callable(executable)
            if not callable(executable):
                raise RuntimePlanError(f"Region {region_id} has a non-callable executable")
            out[region_id] = executable
        return out

    def _build_fast_path(self) -> _FastPath | None:
        """Prebind state for the common case: one resident region, one worker.

        Avoids rebuilding the environment, releasing activations, timing regions,
        and going through backend.execute on every call.
        """
        if self.max_workers != 1 or self._static_order is None or len(self.program.regions) != 1:
            return None
        if self.parameter_store.needs_prefetch:
            return None
        region = self.program.regions[0]
        binding = self.bindings[region.region_id]
        args: list[Any] = [None] * len(region.inputs)
        user_slots: list[tuple[int, int]] = []
        for slot, name in enumerate(region.inputs):
            if name in self.program.user_inputs:
                user_slots.append((slot, self.program.user_inputs.index(name)))
            elif name in self.program.state_bindings:
                args[slot] = self.parameter_store.acquire(name)
            else:
                return None
        if len(user_slots) != len(self.program.user_inputs):
            return None
        if len(region.outputs) != 1:
            return None
        out_name = region.outputs[0]
        if self.program.output_refs != (("value", out_name),):
            return None
        return _FastPath(
            region_id=region.region_id,
            device=binding.device,
            backend_id=binding.backend_id,
            call=self._callables[region.region_id],
            args=args,
            user_slots=tuple(user_slots),
            state_bytes=sum(int(t.numel() * t.element_size()) for t in args if isinstance(t, torch.Tensor)),
        )

    def _verified_static_order(self) -> tuple[Region, ...] | None:
        """Region order to use when one worker runs everything.

        Lowering emits regions in topological order, but rather than trust that, this
        checks it once. When the check fails the dynamic scheduler runs instead, so a
        reordering in the frontend cannot silently produce wrong results.
        """
        seen: set[str] = set()
        for region in self.program.regions:
            if any(dep not in seen for dep in region.depends_on):
                return None
            seen.add(region.region_id)
        return self.program.regions

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
        if not self._run_lock.acquire(blocking=False):
            raise RuntimePlanError(
                "GraphExecutor.run is not reentrant; serialize concurrent calls on one CompiledModule"
            )
        try:
            return self._run_unlocked(flat_inputs)
        finally:
            self._run_lock.release()

    def _run_unlocked(self, flat_inputs: list[Any]) -> tuple[list[Any], ExecutionReport]:
        self.parameter_store.begin_execution()
        if self._fast is not None:
            return self._run_fast(flat_inputs)
        if self._static_resident is not None:
            return self._run_static_resident(flat_inputs)

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

        executor: ThreadPoolExecutor | None = None
        restore_threads: int | None = None
        if self.max_workers > 1 and len(program.regions) > 1:
            executor = inference_thread_pool(max_workers=self.max_workers, thread_name_prefix="streamcompiler-region")
            if self.intraop_threads:
                # Overlapping regions share the cores; the split that won at compile
                # time is applied for this call only and restored afterwards.
                restore_threads = torch.get_num_threads()
                torch.set_num_threads(self.intraop_threads)
        # A single worker following a verified topological order needs none of the
        # readiness bookkeeping, which is pure overhead for small models.
        static_order = self._static_order if executor is None else None

        pending_deps: dict[str, set[str]] = {}
        ready: deque[Region] = deque()
        if static_order is None:
            pending_deps = {r.region_id: set(r.depends_on) for r in program.regions}
            ready = deque(r for r in program.regions if not pending_deps[r.region_id])
            if not ready and program.regions:
                raise RuntimePlanError("Region dependency graph has no entry point (cycle detected)")

        self._prefetch_ahead(-1)
        start_wall = time.perf_counter()

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
                self._track_produced(name, value, device=event.device)
            peak_bytes = max(peak_bytes, activation_bytes)
            self._check_activation_budget(peak_bytes)
            freed, count = self._release_inputs(region, env, remaining, state_ready)
            activation_bytes -= freed
            released += count
            dependents = self._dependents.get(region.region_id, ())
            if static_order is None:
                for dependent in dependents:
                    pending_deps[dependent].discard(region.region_id)
                    if not pending_deps[dependent]:
                        ready.append(program.region_by_id(dependent))
            if self._prefetch_enabled and dependents:
                earliest = min(self._order[d] for d in dependents)
                self._prefetch_ahead(earliest - 1)

        # One inference-mode guard for the whole call: regions never build autograd
        # graphs, and entering the guard per region measurably dominates small ones.
        own_guard = not torch.is_inference_mode_enabled()
        guard = torch.inference_mode() if own_guard else None
        if guard is not None:
            guard.__enter__()
        try:
            if static_order is not None:
                for index, region in enumerate(static_order):
                    args = self._gather_inputs(region, env)
                    # Prefetch the next region only after this one holds its pins so a
                    # tight RAM budget cannot be stolen by speculative staging.
                    if self._prefetch_enabled:
                        self._prefetch_ahead(index)
                    event, outputs = self._run_region(region, args)
                    complete(region, event, outputs)
            while ready or running:
                while ready and (executor is None or len(running) < self.max_workers):
                    region = ready.popleft()
                    args = self._gather_inputs(region, env)
                    if self._prefetch_enabled:
                        self._prefetch_ahead(self._order[region.region_id])
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
            if guard is not None:
                guard.__exit__(None, None, None)
            if executor is not None:
                executor.shutdown(wait=True)
            if restore_threads is not None:
                torch.set_num_threads(restore_threads)

        report.wall_time_s = time.perf_counter() - start_wall
        report.peak_activation_bytes = peak_bytes
        report.released_values = released
        # Nothing can overlap when one worker follows a static order.
        report.parallel_overlaps = 0 if static_order is not None else len(report.overlapping_pairs())
        self.parameter_store.record_compute_intervals([(e.start_s, e.end_s) for e in report.events])
        report.parameter_store = self.parameter_store.stats()
        return self._collect_outputs(env), report

    # ---- helpers ----------------------------------------------------
    def _run_fast(self, flat_inputs: list[Any]) -> tuple[list[Any], ExecutionReport]:
        fast = self._fast
        assert fast is not None
        if len(flat_inputs) != len(self.program.user_inputs):
            raise RuntimePlanError(f"Expected {len(self.program.user_inputs)} flat inputs, received {len(flat_inputs)}")
        args = fast.args
        for slot, user_index in fast.user_slots:
            args[slot] = flat_inputs[user_index]
        start = time.perf_counter()
        already = torch.is_inference_mode_enabled()
        if already:
            result = fast.call(*args)
        else:
            with torch.inference_mode():
                result = fast.call(*args)
        end = time.perf_counter()
        for slot, _ in fast.user_slots:
            args[slot] = None
        outputs = self._coerce_outputs(fast.region_id, result, expected=1)
        peak = outputs[0].numel() * outputs[0].element_size() if isinstance(outputs[0], torch.Tensor) else 0
        self._check_activation_budget(peak)
        report = ExecutionReport(
            wall_time_s=end - start,
            events=[
                RegionEvent(
                    region_id=fast.region_id,
                    device=fast.device,
                    backend_id=fast.backend_id,
                    start_s=start,
                    end_s=end,
                    worker="main",
                )
            ],
            peak_activation_bytes=peak,
            released_values=len(self.program.user_inputs),
            parallel_overlaps=0,
            max_concurrent_regions=1,
            parameter_store=self.parameter_store.stats(),
        )
        return list(outputs), report

    def _run_static_resident(self, flat_inputs: list[Any]) -> tuple[list[Any], ExecutionReport]:
        """Walk a verified region order with state already resident in RAM."""
        plan = self._static_resident
        assert plan is not None
        program = self.program
        env = plan.env
        for name in plan.activation_names:
            env.pop(name, None)
        for name, value in zip(program.user_inputs, flat_inputs, strict=True):
            env[name] = value
        report = ExecutionReport(wall_time_s=0.0)
        peak = 0
        start_wall = time.perf_counter()
        own_guard = not torch.is_inference_mode_enabled()
        guard = torch.inference_mode() if own_guard else None
        if guard is not None:
            guard.__enter__()
        try:
            for region, call, input_names in plan.steps:
                start = time.perf_counter()
                result = call(*(env[name] for name in input_names))
                outputs = self._coerce_outputs(region.region_id, result, expected=len(region.outputs))
                end = time.perf_counter()
                binding = self.bindings[region.region_id]
                report.events.append(
                    RegionEvent(
                        region_id=region.region_id,
                        device=binding.device,
                        backend_id=binding.backend_id,
                        start_s=start,
                        end_s=end,
                        worker="main",
                    )
                )
                for name, value in zip(region.outputs, outputs, strict=True):
                    env[name] = value
                    if isinstance(value, torch.Tensor):
                        peak = max(peak, value.numel() * value.element_size())
                    self._track_produced(name, value, device=binding.device)
                self._check_activation_budget(peak)
        finally:
            if guard is not None:
                guard.__exit__(None, None, None)

        report.wall_time_s = time.perf_counter() - start_wall
        report.peak_activation_bytes = peak
        self.parameter_store.record_compute_intervals([(e.start_s, e.end_s) for e in report.events])
        report.parameter_store = self.parameter_store.stats()
        return self._collect_outputs(env), report

    def _gather_inputs(self, region: Region, env: dict[str, Any]) -> tuple[Any, ...]:
        args: list[Any] = []
        binding = self.bindings[region.region_id]
        for name in region.inputs:
            if name in env:
                args.append(env[name])
            elif name in self.program.state_bindings:
                tensor = self.parameter_store.acquire(name)
                env[name] = tensor
                self.tensor_directory.materialize(
                    name,
                    location=binding.device,
                    tier=MemoryTier.SYSTEM_RAM,
                    nbytes=int(tensor.numel() * tensor.element_size()) if isinstance(tensor, torch.Tensor) else 0,
                    device=binding.device,
                    value=tensor,
                )
                args.append(tensor)
            else:
                raise RuntimePlanError(
                    f"Region {region.region_id} needs value {name} which is not available; "
                    "the execution plan has an invalid dependency order"
                )
        return tuple(args)

    def _run_region(self, region: Region, args: tuple[Any, ...]) -> tuple[RegionEvent, tuple[Any, ...]]:
        binding = self.bindings[region.region_id]
        start = time.perf_counter()
        result = self._callables[region.region_id](*args)
        outputs = coerce_region_result(result)
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

    def _coerce_outputs(self, region_id: str, result: Any, *, expected: int) -> tuple[Any, ...]:
        outputs = coerce_region_result(result)
        if len(outputs) != expected:
            raise RuntimePlanError(f"Region {region_id} produced {len(outputs)} values, plan expects {expected}")
        return outputs

    def _collect_outputs(self, env: dict[str, Any]) -> list[Any]:
        flat_outputs: list[Any] = []
        for kind, ref in self.program.output_refs:
            if kind == "constant":
                flat_outputs.append(ref)
                continue
            name = str(ref)
            if name not in env:
                raise RuntimePlanError(f"Output value {name} was never produced")
            flat_outputs.append(env[name])
        return flat_outputs

    def _release_inputs(
        self,
        region: Region,
        env: dict[str, Any],
        remaining: dict[str, int],
        state_ready: set[str],
    ) -> tuple[int, int]:
        freed = 0
        count = 0
        state_names: list[str] = []
        for name in region.inputs:
            if name not in remaining:
                continue
            left = remaining[name] - 1
            remaining[name] = left
            self.tensor_directory.finish_consumer(name, region_id=region.region_id)
            if left > 0:
                continue
            if name in self.program.state_bindings:
                state_names.append(name)
                state_ready.discard(name)
                self.tensor_directory.release(name)
                continue
            value = env.pop(name, None)
            if isinstance(value, torch.Tensor):
                freed += value.numel() * value.element_size()
            count += 1
            self.tensor_directory.release(name)
        if state_names:
            self.parameter_store.release(tuple(state_names))
        return freed, count

    def _track_produced(self, name: str, value: Any, *, device: str) -> None:
        self.tensor_directory.mark_produced(
            name,
            location=device,
            tier=MemoryTier.SYSTEM_RAM if "cpu" in device or "numa" in device else MemoryTier.DEVICE,
            value=value,
            device=device,
        )
        # Seed consumer counts so release waits for the last async consumer.
        consumers = self._consumers.get(name, 0)
        record = self.tensor_directory.get(name)
        if record is not None:
            record.active_consumers = consumers

    def _prefetch_ahead(self, after_index: int) -> None:
        """Prefetch up to ``prefetch_distance`` regions after ``after_index``.

        Callers pass ``-1`` before the first region and the current region index
        after that region has acquired its parameters. Speculative staging then
        only uses budget that remains once the live region is pinned.
        """
        if not self._prefetch_enabled:
            return
        regions = self.program.regions
        start = after_index + 1
        if start < 0 or start >= len(regions):
            return
        names: list[str] = []
        for region in regions[start : start + self.prefetch_distance]:
            names.extend(region.state_inputs)
        if names:
            self.parameter_store.prefetch(tuple(dict.fromkeys(names)))

    def _prefetch_from(self, index: int) -> None:
        """Compatibility wrapper: prefetch ``index`` and the next distance windows."""
        self._prefetch_ahead(index - 1)
