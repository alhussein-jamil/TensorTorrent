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

import itertools
import os
import sys
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Any

import torch

from streamcompiler.backends.torch_device import coerce_region_result, unwrap_region_callable
from streamcompiler.codegen.regions import Region, RegionBinding, RegionProgram
from streamcompiler.errors import ExecutionCancelled, RuntimePlanError
from streamcompiler.ir.graph import OpCode
from streamcompiler.parallel import inference_thread_pool
from streamcompiler.runtime.activation_spill import SpilledTensor, is_spilled, reload_spilled, spill_tensor
from streamcompiler.runtime.allocation_pool import ActivationAllocator
from streamcompiler.runtime.recompute import NeedsRecompute, is_needs_recompute, mark_for_recompute
from streamcompiler.runtime.schedule import ExecutableSchedule, MemoryTier, PlanInstruction
from streamcompiler.runtime.tensor_directory import TensorDirectory
from streamcompiler.runtime.tensor_store import ParameterStore
from streamcompiler.runtime.transfers import execute_transfer_instruction

# Fork workers inherit this table; keyed by executor instance id.
_FORK_REGION_CALLABLES: dict[int, dict[str, Any]] = {}
_FORK_EXECUTOR_IDS = itertools.count(1)


def _fork_run_region(
    registry_id: int,
    region_id: str,
    device: str,
    backend_id: str,
    args: tuple[Any, ...],
) -> tuple[RegionEvent, tuple[Any, ...]]:
    start = time.perf_counter()
    call = _FORK_REGION_CALLABLES[registry_id][region_id]
    result = call(*args)
    outputs = coerce_region_result(result)
    end = time.perf_counter()
    return (
        RegionEvent(
            region_id=region_id,
            device=device,
            backend_id=backend_id,
            start_s=start,
            end_s=end,
            worker=f"proc-{os.getpid()}",
        ),
        outputs,
    )


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
    released_values: int = 0
    parameter_store_stats: dict[str, Any] = field(default_factory=dict)


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
        buffer_reuse_assignment: dict[str, int] | None = None,
        allow_activation_spill: bool = True,
        activation_overflow_policy: str = "spill",
        process_workers: int = 0,
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
        self._reuse_assignment = dict(buffer_reuse_assignment or {})
        # Buffer-reuse intervals are derived from a sequential schedule. Concurrent
        # region workers can extend a producer's effective lifetime past the
        # sequential last-use, so physical slot reuse is only safe for one worker.
        self._allocator = ActivationAllocator() if self._reuse_assignment and self.max_workers == 1 else None
        self._allow_activation_spill = bool(allow_activation_spill) and activation_budget_bytes is not None
        self._activation_overflow_policy = (
            activation_overflow_policy if activation_overflow_policy in {"spill", "recompute"} else "spill"
        )
        self._spill_events: list[dict[str, Any]] = []
        self._output_producer: dict[str, str] = {
            name: region.region_id for region in program.regions for name in region.outputs
        }
        self._schedule_releases: tuple[PlanInstruction, ...] = ()
        self._schedule_driven = False
        self._process_pool: Any = None
        self._fork_registry_id: int | None = None
        if schedule is not None:
            from streamcompiler.runtime.schedule import ScheduleValidationError, validate_schedule

            violations = validate_schedule(schedule)
            if violations:
                raise RuntimePlanError(
                    f"ExecutableSchedule {schedule.graph_name!r} failed validation: {violations}"
                ) from ScheduleValidationError(str(violations))
            scheduled = [i.executable_ref for i in schedule.compute_ops() if i.executable_ref]
            actual = [r.region_id for r in program.regions]
            if scheduled != actual:
                raise RuntimePlanError(
                    f"ExecutableSchedule compute order {scheduled} does not match program regions {actual}"
                )
            self._schedule_releases = tuple(i for i in schedule.instructions if i.opcode == OpCode.RELEASE)
            self._schedule_driven = True
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
        self._prelude_before: dict[str, tuple[Any, ...]] = {}
        self._cancel_requested = False
        if schedule is not None:
            groups: dict[str, list[Any]] = {}
            instructions = list(schedule.instructions)
            index = 0
            while index < len(instructions):
                inst = instructions[index]
                if inst.opcode == OpCode.TRANSFER:
                    before = str(inst.attributes.get("before_region", ""))
                    group = [inst]
                    if index + 1 < len(instructions) and instructions[index + 1].opcode == OpCode.RECORD_EVENT:
                        index += 1
                        group.append(instructions[index])
                    if index + 1 < len(instructions) and instructions[index + 1].opcode == OpCode.WAIT_EVENT:
                        index += 1
                        group.append(instructions[index])
                    if before:
                        groups.setdefault(before, []).extend(group)
                index += 1
            self._prelude_before = {k: tuple(v) for k, v in groups.items()}
        self._init_process_workers(int(process_workers))

    def _init_process_workers(self, process_workers: int) -> None:
        """Attach a fork process pool when requested (Linux) for concurrent regions."""
        if process_workers <= 0 or self.max_workers <= 1:
            return
        if sys.platform != "linux":
            return
        from streamcompiler.runtime.process_workers import ProcessWorkerPool

        self._fork_registry_id = next(_FORK_EXECUTOR_IDS)
        _FORK_REGION_CALLABLES[self._fork_registry_id] = dict(self._callables)
        self._process_pool = ProcessWorkerPool(
            max_workers=min(process_workers, self.max_workers),
            start_method="fork",
            warm_up=True,
        )

    def close(self) -> None:
        pool = self._process_pool
        self._process_pool = None
        if pool is not None:
            pool.shutdown(wait=True)
        if self._fork_registry_id is not None:
            _FORK_REGION_CALLABLES.pop(self._fork_registry_id, None)
            self._fork_registry_id = None

    def _check_activation_budget(self, peak_bytes: int, env: dict[str, Any] | None = None) -> int:
        """Enforce the host activation budget, spilling cold tensors when allowed.

        Returns the (possibly reduced) live activation byte count after any spill.
        """
        budget = self.activation_budget_bytes
        if budget is None or peak_bytes <= budget:
            return peak_bytes
        if not self._allow_activation_spill or env is None:
            raise RuntimePlanError(f"Peak live activations {peak_bytes} bytes exceed activation_budget_bytes={budget}")
        live = peak_bytes
        while live > budget:
            candidate = self._pick_spill_candidate(env)
            if candidate is None:
                raise RuntimePlanError(
                    f"Peak live activations {live} bytes exceed activation_budget_bytes={budget} "
                    "and no cold activation remains to spill"
                )
            name, value = candidate
            producer = self._output_producer.get(name, "")
            if self._activation_overflow_policy == "recompute" and producer:
                env[name] = mark_for_recompute(value, producer_region_id=producer)
                live -= int(value.numel() * value.element_size())
                self._spill_events.append({"event": "recompute_drop", "name": name, "producer_region_id": producer})
            else:
                spilled = spill_tensor(value)
                env[name] = spilled
                live -= spilled.nbytes
                self._spill_events.append({"event": "spill", "name": name, **spilled.as_dict()})
            self.tensor_directory.release(name)
        return live

    def _pick_spill_candidate(self, env: dict[str, Any]) -> tuple[str, torch.Tensor] | None:
        """Largest live host activation that is not a user input, state, or final output."""
        protected = set(self.program.user_inputs)
        protected.update(self.program.state_bindings)
        for kind, ref in self.program.output_refs:
            if kind == "value":
                protected.add(str(ref))
        best: tuple[str, torch.Tensor] | None = None
        best_bytes = -1
        for name, value in env.items():
            if name in protected or not isinstance(value, torch.Tensor):
                continue
            nbytes = int(value.numel() * value.element_size())
            if nbytes > best_bytes:
                best = (name, value)
                best_bytes = nbytes
        return best

    def _materialize_env_value(self, name: str, env: dict[str, Any]) -> Any:
        value = env.get(name)
        if is_spilled(value):
            assert isinstance(value, SpilledTensor)
            tensor = reload_spilled(value)
            env[name] = tensor
            self._spill_events.append({"event": "reload", "name": name, "nbytes": value.nbytes})
            return tensor
        if is_needs_recompute(value):
            assert isinstance(value, NeedsRecompute)
            producer_id = value.producer_region_id
            producer = self.program.region_by_id(producer_id)
            args = []
            for input_name in producer.inputs:
                if input_name not in env and input_name in self.program.state_bindings:
                    env[input_name] = self.parameter_store.acquire(input_name)
                if input_name not in env:
                    raise RuntimePlanError(f"Cannot recompute {name} via {producer_id}: missing input {input_name}")
                args.append(self._materialize_env_value(input_name, env))
            result = self._callables[producer_id](*args)
            outputs = self._coerce_outputs(producer_id, result, expected=len(producer.outputs))
            for out_name, out_value in zip(producer.outputs, outputs, strict=True):
                env[out_name] = out_value
            self._spill_events.append({"event": "recompute", "name": name, "producer_region_id": producer_id})
            return env[name]
        return value

    def _place_activation(self, name: str, value: Any) -> Any:
        """Optionally copy ``value`` into a reuse-pool slot so non-overlapping ids share storage."""
        if self._allocator is None or not isinstance(value, torch.Tensor):
            return value
        slot = self._reuse_assignment.get(name)
        if slot is None:
            return value
        return self._allocator.acquire(slot, name, value)

    def _fire_due_schedule_releases(
        self,
        completed_computes: set[str],
        pending_releases: list[PlanInstruction],
        env: dict[str, Any],
    ) -> tuple[int, int]:
        """Execute every schedule Release whose Compute dependencies have finished."""
        freed = 0
        count = 0
        still_pending: list[PlanInstruction] = []
        for rel in pending_releases:
            if not all(dep in completed_computes for dep in rel.depends_on):
                still_pending.append(rel)
                continue
            producer_id = str(rel.attributes.get("producer_region") or "")
            if not producer_id and rel.inputs:
                raw = rel.inputs[0]
                producer_id = raw.split("activation::", 1)[-1] if raw.startswith("activation::") else ""
            if not producer_id or producer_id not in self._order:
                continue
            producer = self.program.region_by_id(producer_id)
            for name in producer.outputs:
                if name in self.program.user_inputs or name in self.program.state_bindings:
                    continue
                value = env.pop(name, None)
                if self._allocator is not None:
                    slot = self._reuse_assignment.get(name)
                    if slot is not None:
                        self._allocator.release(slot)
                if isinstance(value, torch.Tensor):
                    freed += int(value.numel() * value.element_size())
                    count += 1
                elif is_spilled(value):
                    assert isinstance(value, SpilledTensor)
                    value.path.unlink(missing_ok=True)
                    freed += value.nbytes
                    count += 1
                self.tensor_directory.release(name)
        pending_releases[:] = still_pending
        return freed, count

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
            released_values=len(self.program.user_inputs),
            parameter_store_stats=self.parameter_store.stats(),
        )

    def _verified_static_order(self) -> tuple[Region, ...] | None:
        """Region order to use when one worker runs everything.

        When an ExecutableSchedule is present its Compute opcodes are the
        authoritative order (already validated against program.regions).
        Otherwise lowering order is checked for topological validity.
        """
        if self.schedule is not None:
            ordered: list[Region] = []
            seen: set[str] = set()
            for inst in self.schedule.compute_ops():
                rid = str(inst.executable_ref)
                region = self.program.region_by_id(rid)
                if any(dep not in seen for dep in region.depends_on):
                    return None
                ordered.append(region)
                seen.add(rid)
            return tuple(ordered)
        seen = set()
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

    def request_cancel(self) -> None:
        """Ask the in-flight ``run`` to abort at the next region boundary.

        Safe to call from another thread. A single-region fast path can only
        observe the flag before the kernel starts; once that call is inside
        the region executable it runs to completion. Multi-region paths stop
        submitting new work, wait for already-running workers, release partial
        activations / parameter pins, then raise :class:`ExecutionCancelled`.
        """
        self._cancel_requested = True

    def _raise_if_cancelled(
        self,
        env: dict[str, Any] | None = None,
        *,
        keep_resident_state: bool = False,
    ) -> None:
        if not self._cancel_requested:
            return
        if env is not None:
            self._release_partial_env(env, keep_resident_state=keep_resident_state)
        self._cancel_requested = False
        raise ExecutionCancelled("GraphExecutor.run was cancelled")

    def _release_partial_env(self, env: dict[str, Any], *, keep_resident_state: bool = False) -> None:
        state_names = [name for name in env if name in self.program.state_bindings]
        for name in list(env):
            if name in self.program.user_inputs:
                continue
            if keep_resident_state and name in self.program.state_bindings:
                continue
            self.tensor_directory.release(name)
            env.pop(name, None)
        if state_names and not keep_resident_state:
            self.parameter_store.release(tuple(state_names))

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
        if self._fast is not None:
            return self._run_fast(flat_inputs)
        self._spill_events.clear()
        self.parameter_store.begin_execution()
        self._transfer_events.clear()
        from streamcompiler.runtime.async_events import EventRegistry

        self._event_registry = EventRegistry()
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
        completed_computes: set[str] = set()
        pending_releases: list[PlanInstruction] = list(self._schedule_releases)

        for name, value in zip(program.user_inputs, flat_inputs, strict=True):
            env[name] = value

        executor: ThreadPoolExecutor | None = None
        restore_threads: int | None = None
        use_process_pool = self._process_pool is not None and self.max_workers > 1 and len(program.regions) > 1
        if self.max_workers > 1 and len(program.regions) > 1 and not use_process_pool:
            executor = inference_thread_pool(max_workers=self.max_workers, thread_name_prefix="streamcompiler-region")
            if self.intraop_threads:
                # Overlapping regions share the cores; the split that won at compile
                # time is applied for this call only and restored afterwards.
                restore_threads = torch.get_num_threads()
                torch.set_num_threads(self.intraop_threads)
        # A single worker following a verified topological order needs none of the
        # readiness bookkeeping, which is pure overhead for small models.
        # Schedule Compute order is preferred when present (_verified_static_order).
        static_order = self._static_order if (executor is None and not use_process_pool) else None

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
                value = self._place_activation(name, value)
                env[name] = value
                if isinstance(value, torch.Tensor):
                    activation_bytes += value.numel() * value.element_size()
                self._track_produced(name, value, device=event.device)
            peak_bytes = max(peak_bytes, activation_bytes)
            activation_bytes = self._check_activation_budget(activation_bytes, env)
            peak_bytes = max(peak_bytes, activation_bytes)
            if self._schedule_driven:
                # State pins still use consumer counts; activations follow Release ops.
                freed_state, _ = self._release_inputs(region, env, remaining, state_ready, activations=False)
                activation_bytes -= freed_state
                completed_computes.add(f"compute::{region.region_id}")
                freed, count = self._fire_due_schedule_releases(completed_computes, pending_releases, env)
            else:
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
                    self._raise_if_cancelled(env)
                    args = self._gather_inputs(region, env)
                    # Prefetch the next region only after this one holds its pins so a
                    # tight RAM budget cannot be stolen by speculative staging.
                    if self._prefetch_enabled:
                        self._prefetch_ahead(index)
                    event, outputs = self._run_region(region, args)
                    complete(region, event, outputs)
            while ready or running:
                self._raise_if_cancelled(env)
                while ready and ((executor is None and not use_process_pool) or len(running) < self.max_workers):
                    if self._cancel_requested:
                        break
                    region = ready.popleft()
                    args = self._gather_inputs(region, env)
                    if self._prefetch_enabled:
                        self._prefetch_ahead(self._order[region.region_id])
                    if use_process_pool:
                        binding = self.bindings[region.region_id]
                        assert self._process_pool is not None and self._fork_registry_id is not None
                        running[
                            self._process_pool.submit(
                                _fork_run_region,
                                self._fork_registry_id,
                                region.region_id,
                                binding.device,
                                binding.backend_id,
                                args,
                            )
                        ] = region
                    elif executor is None:
                        event, outputs = self._run_region(region, args)
                        complete(region, event, outputs)
                    else:
                        running[executor.submit(self._run_region, region, args)] = region
                if self._cancel_requested and not running:
                    self._raise_if_cancelled(env)
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

        self._cancel_requested = False
        report.wall_time_s = time.perf_counter() - start_wall
        report.peak_activation_bytes = peak_bytes
        report.released_values = released
        # Nothing can overlap when one worker follows a static order.
        report.parallel_overlaps = 0 if static_order is not None else len(report.overlapping_pairs())
        self.parameter_store.record_compute_intervals([(e.start_s, e.end_s) for e in report.events])
        stats = self.parameter_store.stats()
        if isinstance(stats, dict):
            stats = dict(stats)
            stats["transfer_event_count"] = len(self._transfer_events)
            stats["spill_event_count"] = len(self._spill_events)
            stats["schedule_driven"] = self._schedule_driven
            stats["tensor_directory_live"] = sum(
                1 for rec in self.tensor_directory.snapshot().values() if rec.get("state") != "released"
            )
            if "exposed_io_s" in stats:
                stats["exposed_stall_s"] = stats["exposed_io_s"]
        report.parameter_store = stats
        return self._collect_outputs(env), report

    # ---- helpers ----------------------------------------------------
    def _run_fast(self, flat_inputs: list[Any]) -> tuple[list[Any], ExecutionReport]:
        fast = self._fast
        assert fast is not None
        self._raise_if_cancelled()
        n_user = len(self.program.user_inputs)
        if len(flat_inputs) != n_user:
            raise RuntimePlanError(f"Expected {n_user} flat inputs, received {len(flat_inputs)}")
        args = fast.args
        for slot, user_index in fast.user_slots:
            args[slot] = flat_inputs[user_index]
        start = time.perf_counter()
        if torch.is_inference_mode_enabled():
            result = fast.call(*args)
        else:
            with torch.inference_mode():
                result = fast.call(*args)
        end = time.perf_counter()
        for slot, _ in fast.user_slots:
            args[slot] = None
        # Fast path guarantees a single output; skip coerce for the common Tensor case.
        if isinstance(result, torch.Tensor):
            outputs: list[Any] = [result]
            peak = result.numel() * result.element_size()
        else:
            coerced = self._coerce_outputs(fast.region_id, result, expected=1)
            outputs = list(coerced)
            out0 = outputs[0]
            peak = out0.numel() * out0.element_size() if isinstance(out0, torch.Tensor) else 0
        self._check_activation_budget(peak)
        self._cancel_requested = False
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
            released_values=fast.released_values,
            parallel_overlaps=0,
            max_concurrent_regions=1,
            parameter_store=fast.parameter_store_stats,
        )
        return outputs, report

    def _run_static_resident(self, flat_inputs: list[Any]) -> tuple[list[Any], ExecutionReport]:
        """Walk schedule/program region order with state already resident in RAM."""
        plan = self._static_resident
        assert plan is not None
        program = self.program
        env = plan.env
        for name in plan.activation_names:
            value = env.pop(name, None)
            if is_spilled(value):
                assert isinstance(value, SpilledTensor)
                value.path.unlink(missing_ok=True)
        for name, value in zip(program.user_inputs, flat_inputs, strict=True):
            env[name] = value
        report = ExecutionReport(wall_time_s=0.0)
        peak = 0
        live = 0
        released = 0
        completed_computes: set[str] = set()
        pending_releases: list[PlanInstruction] = list(self._schedule_releases)
        start_wall = time.perf_counter()
        own_guard = not torch.is_inference_mode_enabled()
        guard = torch.inference_mode() if own_guard else None
        if guard is not None:
            guard.__enter__()
        try:
            for region, call, input_names in plan.steps:
                self._raise_if_cancelled(env, keep_resident_state=True)
                self._run_scheduled_transfers_before(region, env)
                args_list: list[Any] = []
                for name in input_names:
                    if name not in env:
                        raise RuntimePlanError(f"Region {region.region_id} needs value {name} which is not available")
                    args_list.append(self._materialize_env_value(name, env))
                start = time.perf_counter()
                result = call(*args_list)
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
                    value = self._place_activation(name, value)
                    env[name] = value
                    if isinstance(value, torch.Tensor):
                        live += int(value.numel() * value.element_size())
                    self._track_produced(name, value, device=binding.device)
                peak = max(peak, live)
                live = self._check_activation_budget(live, env)
                peak = max(peak, live)
                if self._schedule_driven:
                    completed_computes.add(f"compute::{region.region_id}")
                    freed, count = self._fire_due_schedule_releases(completed_computes, pending_releases, env)
                    live -= freed
                    released += count
        finally:
            if guard is not None:
                guard.__exit__(None, None, None)

        self._cancel_requested = False
        report.wall_time_s = time.perf_counter() - start_wall
        report.peak_activation_bytes = peak
        report.released_values = released
        self.parameter_store.record_compute_intervals([(e.start_s, e.end_s) for e in report.events])
        stats = self.parameter_store.stats()
        if isinstance(stats, dict):
            stats = dict(stats)
            stats["transfer_event_count"] = len(self._transfer_events)
            stats["spill_event_count"] = len(self._spill_events)
            stats["schedule_driven"] = self._schedule_driven
            stats["tensor_directory_live"] = sum(
                1 for rec in self.tensor_directory.snapshot().values() if rec.get("state") != "released"
            )
            if "exposed_io_s" in stats:
                stats["exposed_stall_s"] = stats["exposed_io_s"]
        report.parameter_store = stats
        return self._collect_outputs(env), report

    def _run_scheduled_transfers_before(self, region: Region, env: dict[str, Any]) -> None:
        """Execute Transfer → RecordEvent → WaitEvent prelude scheduled before ``region``.

        RecordEvent stores a named handle in the per-run event registry; WaitEvent
        waits that same handle. Device DMA may be simulated when hardware is absent.
        """
        from streamcompiler.runtime.async_events import EventRegistry, make_event

        registry = getattr(self, "_event_registry", None)
        if registry is None:
            registry = EventRegistry()
            self._event_registry = registry

        pending = self._prelude_before.get(region.region_id)
        if not pending:
            return

        for inst in pending:
            if inst.opcode == OpCode.WAIT_EVENT:
                waits_for = str(inst.attributes.get("waits_for") or "")
                event = registry.get(waits_for)
                start = time.perf_counter()
                event.wait()
                end = time.perf_counter()
                self._transfer_events.append(
                    {
                        "name": inst.name,
                        "start_s": start,
                        "end_s": end,
                        "resource": inst.resource,
                        "backend": "cuda_event" if event.cuda_event is not None else "wait_event",
                        "nbytes": 0,
                        "simulated": bool(inst.attributes.get("simulated_until_validated", False))
                        and event.cuda_event is None,
                        "elided": False,
                        "notes": (
                            "CUDA event wait"
                            if event.cuda_event is not None
                            else f"WaitEvent on RecordEvent {waits_for!r}"
                        ),
                    }
                )
                continue

            if inst.opcode == OpCode.RECORD_EVENT:
                event = make_event(inst.name, str(inst.resource))
                start = time.perf_counter()
                event.record()
                end = time.perf_counter()
                registry.store(inst.name, event)
                self._transfer_events.append(
                    {
                        "name": inst.name,
                        "start_s": start,
                        "end_s": end,
                        "resource": inst.resource,
                        "backend": "cuda_event" if event.cuda_event is not None else "record_event",
                        "nbytes": 0,
                        "simulated": bool(inst.attributes.get("simulated_until_validated", False))
                        and event.cuda_event is None,
                        "elided": False,
                        "notes": f"RecordEvent {inst.name!r}",
                    }
                )
                continue

            if inst.opcode != OpCode.TRANSFER:
                continue
            value_name = inst.inputs[0] if inst.inputs else None
            if value_name is None:
                continue
            # Map synthetic activation::region ids onto real env values when possible.
            value = env.get(value_name)
            if value is None and value_name.startswith("activation::"):
                producer = value_name.split("::", 1)[1]
                for name in self.program.region_by_id(producer).outputs:
                    if name in env:
                        value = env[name]
                        value_name = name
                        break
            if value is None:
                continue
            # Schedule uses synthetic ``activation::region`` ids; directory tracks
            # the real env value names produced by Compute.
            from dataclasses import replace

            xfer_inst = inst
            if value_name != (inst.inputs[0] if inst.inputs else None):
                xfer_inst = replace(inst, inputs=(value_name,))
            start = time.perf_counter()
            out, result = execute_transfer_instruction(xfer_inst, value, self.tensor_directory)
            end = time.perf_counter()
            if out is not value and value_name in env:
                env[value_name] = out
            self._transfer_events.append(
                {
                    "name": inst.name,
                    "start_s": start,
                    "end_s": end,
                    "resource": inst.resource,
                    "backend": result.backend,
                    "nbytes": result.nbytes,
                    "simulated": result.simulated,
                    "elided": result.backend == "elided_duplicate",
                    "notes": result.notes,
                }
            )

    def _gather_inputs(self, region: Region, env: dict[str, Any]) -> tuple[Any, ...]:
        self._run_scheduled_transfers_before(region, env)
        args: list[Any] = []
        binding = self.bindings[region.region_id]
        for name in region.inputs:
            if name in env:
                args.append(self._materialize_env_value(name, env))
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
        *,
        activations: bool = True,
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
            if not activations:
                continue
            value = env.pop(name, None)
            if self._allocator is not None:
                slot = self._reuse_assignment.get(name)
                if slot is not None:
                    self._allocator.release(slot)
            if isinstance(value, torch.Tensor):
                freed += value.numel() * value.element_size()
            elif is_spilled(value):
                assert isinstance(value, SpilledTensor)
                value.path.unlink(missing_ok=True)
                freed += value.nbytes
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
