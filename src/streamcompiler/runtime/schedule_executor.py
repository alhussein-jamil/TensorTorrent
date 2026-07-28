"""Instruction-DAG executor: ExecutableSchedule is the exclusive runtime program.

Every Prefetch / Load / Transfer / RecordEvent / WaitEvent / Compute / Evict /
Release op is dispatched when its ``depends_on`` instructions have completed.
Independent instructions may overlap; compute order need not match region order.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Any

import torch

from streamcompiler.backends.torch_device import coerce_region_result
from streamcompiler.codegen.regions import RegionBinding, RegionProgram
from streamcompiler.errors import ExecutionCancelled, RuntimePlanError
from streamcompiler.ir.graph import OpCode
from streamcompiler.runtime.copies import CopyStore
from streamcompiler.runtime.schedule import ExecutableSchedule, PlanInstruction
from streamcompiler.runtime.streams import DeviceStreams, EventRegistry, StreamEvent
from streamcompiler.runtime.tensor_store import ParameterStore
from streamcompiler.runtime.transfers import select_transfer_backend


@dataclass
class InstructionEvent:
    name: str
    opcode: str
    resource: str
    submitted_s: float
    start_s: float
    end_s: float
    nbytes: int = 0
    notes: str = ""
    prefetch_hit: bool | None = None
    exposed_stall_s: float = 0.0
    enqueue_start_s: float = 0.0
    enqueue_end_s: float = 0.0
    complete_s: float = 0.0
    consumer_wait_s: float = 0.0
    simulated: bool = False

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)


@dataclass
class ScheduleReport:
    wall_time_s: float
    events: list[InstructionEvent] = field(default_factory=list)
    parallel_overlaps: int = 0
    max_concurrent: int = 1
    copy_snapshot: dict[str, Any] = field(default_factory=dict)
    parameter_store: dict[str, Any] = field(default_factory=dict)
    multi_copy_peaks: list[dict[str, Any]] = field(default_factory=list)
    peak_activation_bytes: int = 0

    def overlapping_pairs(self) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = []
        ordered = sorted(self.events, key=lambda e: e.start_s)
        for i, first in enumerate(ordered):
            for second in ordered[i + 1 :]:
                if second.start_s >= first.end_s:
                    break
                if (
                    first.opcode == "Compute"
                    and second.opcode == "Compute"
                    or {first.opcode, second.opcode} & {"Transfer", "Compute", "Prefetch", "Load"}
                ):
                    pairs.append((first.name, second.name))
        return pairs

    def as_dict(self) -> dict[str, Any]:
        return {
            "wall_time_s": self.wall_time_s,
            "instruction_count": len(self.events),
            "parallel_overlaps": self.parallel_overlaps,
            "max_concurrent": self.max_concurrent,
            "copy_snapshot": self.copy_snapshot,
            "multi_copy_peaks": self.multi_copy_peaks,
            "peak_activation_bytes": self.peak_activation_bytes,
            "parameter_store": self.parameter_store,
            "instructions": [
                {
                    "name": e.name,
                    "opcode": e.opcode,
                    "resource": e.resource,
                    "duration_s": e.duration_s,
                    "nbytes": e.nbytes,
                    "prefetch_hit": e.prefetch_hit,
                    "exposed_stall_s": e.exposed_stall_s,
                    "notes": e.notes,
                }
                for e in self.events
            ],
        }


class ScheduleExecutor:
    """Runs an :class:`ExecutableSchedule` as an instruction dependency DAG."""

    def __init__(
        self,
        program: RegionProgram,
        bindings: dict[str, RegionBinding],
        schedule: ExecutableSchedule,
        *,
        parameter_store: ParameterStore,
        streams: DeviceStreams | None = None,
        max_inflight: int = 8,
        process_pool: Any | None = None,
        fork_registry_id: int | None = None,
        callables: dict[str, Any] | None = None,
        allocator: Any | None = None,
        activation_budget_bytes: int | None = None,
        allow_activation_spill: bool = False,
        spill_events: list[dict[str, Any]] | None = None,
        reuse_assignment: dict[str, int] | None = None,
    ) -> None:
        from streamcompiler.runtime.schedule import ScheduleValidationError, validate_schedule

        violations = validate_schedule(schedule)
        if violations:
            raise RuntimePlanError(
                f"ExecutableSchedule {schedule.graph_name!r} failed validation: {violations}"
            ) from ScheduleValidationError(str(violations))
        self.program = program
        self.bindings = bindings
        self.schedule = schedule
        self.parameter_store = parameter_store
        self.streams = streams if streams is not None else DeviceStreams()
        self.max_inflight = max(1, int(max_inflight))
        self.process_pool = process_pool
        self.fork_registry_id = fork_registry_id
        self.allocator = allocator
        self.activation_budget_bytes = activation_budget_bytes
        self.allow_activation_spill = bool(allow_activation_spill)
        self._spill_events = spill_events if spill_events is not None else []
        self._reuse_assignment = dict(reuse_assignment or {})
        self.copies = CopyStore()
        self._by_name = {i.name: i for i in schedule.instructions}
        self._dependents: dict[str, list[str]] = defaultdict(list)
        for inst in schedule.instructions:
            for dep in inst.depends_on:
                self._dependents[dep].append(inst.name)
        if callables is not None:
            self._callables = callables
        else:
            self._callables = {
                rid: getattr(binding.compiled, "executable", binding.compiled) for rid, binding in bindings.items()
            }
        self._run_lock = threading.Lock()
        self._cancel = False
        self._closed = False
        self._pending_transfers: dict[tuple[str, str], Future[Any]] = {}
        self._transfer_lock = threading.Lock()
        self._multi_copy_peaks: list[dict[str, Any]] = []
        self._sync_pool = ThreadPoolExecutor(
            max_workers=max(4, self.max_inflight),
            thread_name_prefix="schedule-sync",
        )

    def request_cancel(self) -> None:
        self._cancel = True

    def close(self) -> None:
        self._closed = True
        self._cancel = True
        with self._transfer_lock:
            self._pending_transfers.clear()
        self._sync_pool.shutdown(wait=True, cancel_futures=True)
        self.streams.shutdown(wait=True)

    def run(self, flat_inputs: list[Any]) -> tuple[list[Any], ScheduleReport]:
        if self._closed:
            raise RuntimePlanError("ScheduleExecutor is closed")
        if not self._run_lock.acquire(blocking=False):
            raise RuntimePlanError("ScheduleExecutor.run is not reentrant")
        try:
            return self._run_unlocked(flat_inputs)
        finally:
            self._run_lock.release()

    def _run_unlocked(self, flat_inputs: list[Any]) -> tuple[list[Any], ScheduleReport]:
        if self._closed:
            raise RuntimePlanError("ScheduleExecutor is closed")
        if self._cancel:
            self._cancel = False
            raise ExecutionCancelled("Schedule execution cancelled")
        self._cancel = False
        with self._transfer_lock:
            self._pending_transfers.clear()
        self.copies = CopyStore()
        self._multi_copy_peaks = []
        registry = EventRegistry()
        report = ScheduleReport(wall_time_s=0.0)
        events_by_name: dict[str, InstructionEvent] = {}
        completed: set[str] = set()
        remaining_deps: dict[str, set[str]] = {inst.name: set(inst.depends_on) for inst in self.schedule.instructions}
        ready: deque[str] = deque(name for name, deps in remaining_deps.items() if not deps)
        running: dict[Future[Any], str] = {}
        # Seed user inputs onto their host resource (first CPU compute resource or cpu).
        host = self._default_host_resource()
        if len(flat_inputs) != len(self.program.user_inputs):
            raise RuntimePlanError(f"Expected {len(self.program.user_inputs)} inputs, got {len(flat_inputs)}")
        for name, value in zip(self.program.user_inputs, flat_inputs, strict=True):
            self.copies.put(name, host, value, tier="system_ram")
            # Alias common host labels so residency Transfer sources resolve.
            if host != "cpu":
                self.copies.put(name, "cpu", value, tier="system_ram")
            if host != "host":
                self.copies.put(name, "host", value, tier="system_ram")

        self.parameter_store.begin_execution()
        wall0 = time.perf_counter()

        def _finish(name: str, event: InstructionEvent) -> None:
            events_by_name[name] = event
            report.events.append(event)
            completed.add(name)
            for child in self._dependents.get(name, ()):
                remaining_deps[child].discard(name)
                if (
                    not remaining_deps[child]
                    and child not in completed
                    and child not in running.values()
                    and child not in ready
                ):
                    ready.append(child)

        while ready or running:
            if self._cancel and not running:
                self._cancel = False
                raise ExecutionCancelled("Schedule execution cancelled")
            while ready and len(running) < self.max_inflight and not self._cancel:
                name = ready.popleft()
                if name in completed:
                    continue
                inst = self._by_name[name]
                submitted = time.perf_counter()
                fut = self._dispatch(inst, registry, host, submitted)
                running[fut] = name
                report.max_concurrent = max(report.max_concurrent, len(running))
            if not running:
                if self._cancel:
                    self._cancel = False
                    raise ExecutionCancelled("Schedule execution cancelled")
                if ready:
                    continue
                break
            done, _ = wait(list(running), return_when=FIRST_COMPLETED)
            for fut in done:
                name = running.pop(fut)
                event = fut.result()
                _finish(name, event)
            if self._cancel and not running:
                self._cancel = False
                raise ExecutionCancelled("Schedule execution cancelled")

        # Any instruction never completed → dependency bug.
        missing = [i.name for i in self.schedule.instructions if i.name not in completed]
        if missing:
            raise RuntimePlanError(f"Schedule left unfinished instructions: {missing}")

        report.wall_time_s = time.perf_counter() - wall0
        report.parallel_overlaps = len(report.overlapping_pairs())
        report.copy_snapshot = self.copies.snapshot()
        report.multi_copy_peaks = list(getattr(self, "_multi_copy_peaks", []))
        report.peak_activation_bytes = self.copies.peak_bytes()
        compute_intervals = [(e.start_s, e.end_s) for e in report.events if e.opcode == "Compute"]
        if hasattr(self.parameter_store, "record_compute_intervals"):
            self.parameter_store.record_compute_intervals(compute_intervals)
        stats = self.parameter_store.stats()
        if isinstance(stats, dict):
            stats = dict(stats)
            stats["schedule_instruction_events"] = len(report.events)
            stats["schedule_driven"] = True
            stats["peak_activation_bytes"] = report.peak_activation_bytes
        report.parameter_store = stats if isinstance(stats, dict) else {}
        return self._collect_outputs(host), report

    def _default_host_resource(self) -> str:
        for binding in self.bindings.values():
            if "cpu" in binding.device or "numa" in binding.device:
                return binding.device
        return "cpu"

    def _dispatch(
        self,
        inst: PlanInstruction,
        registry: EventRegistry,
        host: str,
        submitted: float,
    ) -> Future[Any]:
        opcode = inst.opcode
        if opcode == OpCode.PREFETCH:
            return self._submit_sync(lambda: self._exec_prefetch(inst, submitted))
        if opcode == OpCode.LOAD:
            return self._submit_sync(lambda: self._exec_load(inst, submitted))
        if opcode == OpCode.TRANSFER:
            return self._submit_transfer(inst, registry, submitted)
        if opcode == OpCode.RECORD_EVENT:
            return self._submit_sync(lambda: self._exec_record(inst, registry, submitted))
        if opcode == OpCode.WAIT_EVENT:
            return self._submit_sync(lambda: self._exec_wait(inst, registry, submitted))
        if opcode == OpCode.COMPUTE:
            return self._submit_compute(inst, submitted)
        if opcode == OpCode.RELEASE:
            return self._submit_sync(lambda: self._exec_release(inst, submitted))
        if opcode == OpCode.EVICT:
            return self._submit_sync(lambda: self._exec_evict(inst, submitted))
        raise RuntimePlanError(f"Unsupported schedule opcode {opcode}")

    def _submit_sync(self, fn: Any) -> Future[Any]:
        if self._closed:
            fut: Future[Any] = Future()
            fut.set_exception(RuntimePlanError("ScheduleExecutor is closed"))
            return fut
        return self._sync_pool.submit(fn)

    def _exec_prefetch(self, inst: PlanInstruction, submitted: float) -> InstructionEvent:
        start = time.perf_counter()
        tensor_id = inst.inputs[0] if inst.inputs else ""
        hit = False
        nbytes = inst.nbytes
        # Prefetch = async stage into parameter store cache when streaming.
        if tensor_id and getattr(self.parameter_store, "needs_prefetch", False):
            # Map state::region or env name onto store keys.
            names = self._state_env_names(inst)
            try:
                self.parameter_store.prefetch(tuple(names))
            except Exception as exc:  # budget / closed — Load will materialize
                end = time.perf_counter()
                return InstructionEvent(
                    name=inst.name,
                    opcode=inst.opcode.value,
                    resource=str(inst.resource),
                    submitted_s=submitted,
                    start_s=start,
                    end_s=end,
                    nbytes=nbytes,
                    prefetch_hit=False,
                    notes=f"schedule Prefetch skipped: {exc}",
                )
        end = time.perf_counter()
        return InstructionEvent(
            name=inst.name,
            opcode=inst.opcode.value,
            resource=str(inst.resource),
            submitted_s=submitted,
            start_s=start,
            end_s=end,
            nbytes=nbytes,
            prefetch_hit=hit,
            notes="schedule Prefetch",
        )

    def _exec_load(self, inst: PlanInstruction, submitted: float) -> InstructionEvent:
        start = time.perf_counter()
        stall0 = time.perf_counter()
        names = self._state_env_names(inst)
        nbytes = 0
        dest = str(inst.destination or inst.resource)
        for env_name in names:
            tensor = self.parameter_store.acquire(env_name)
            # Store under both env name and logical target if distinct.
            target = self.program.state_bindings.get(env_name, env_name)
            self.copies.put(env_name, dest, tensor, tier="system_ram")
            if target != env_name:
                self.copies.put(target, dest, tensor, tier="system_ram")
            if isinstance(tensor, torch.Tensor):
                nbytes += int(tensor.numel() * tensor.element_size())
        stall = time.perf_counter() - stall0
        end = time.perf_counter()
        return InstructionEvent(
            name=inst.name,
            opcode=inst.opcode.value,
            resource=str(inst.resource),
            submitted_s=submitted,
            start_s=start,
            end_s=end,
            nbytes=nbytes or inst.nbytes,
            exposed_stall_s=stall,
            notes="schedule Load",
        )

    def _state_env_names(self, inst: PlanInstruction) -> list[str]:
        region_id = str(inst.attributes.get("region_id") or "")
        if region_id and region_id in self.bindings:
            return list(self.bindings[region_id].region.state_inputs)
        # Fall back: inputs named state::region or env names already.
        out: list[str] = []
        for raw in inst.inputs:
            if raw.startswith("state::"):
                rid = raw.split("::", 1)[1]
                if rid in self.bindings:
                    out.extend(self.bindings[rid].region.state_inputs)
            elif raw in self.program.state_bindings:
                out.append(raw)
        return out

    def _submit_transfer(self, inst: PlanInstruction, registry: EventRegistry, submitted: float) -> Future[Any]:
        tensor_id = inst.inputs[0] if inst.inputs else ""
        dest = str(inst.destination or inst.resource)
        src = str(inst.source or self._default_host_resource())
        key = (tensor_id, dest)

        def _body() -> InstructionEvent:
            enqueue_start = time.perf_counter()
            with self._transfer_lock:
                existing = self._pending_transfers.get(key)
            if existing is not None and not existing.done():
                # Another transfer to same dest is in flight — join by sharing future.
                inst.attributes["_async_future"] = existing
                inst.attributes["_enqueue_start_s"] = enqueue_start
                inst.attributes["_enqueue_end_s"] = time.perf_counter()
                end = time.perf_counter()
                return InstructionEvent(
                    name=inst.name,
                    opcode=inst.opcode.value,
                    resource=str(inst.resource),
                    submitted_s=submitted,
                    start_s=enqueue_start,
                    end_s=end,
                    nbytes=0,
                    notes="joined in-progress transfer (async)",
                    enqueue_start_s=enqueue_start,
                    enqueue_end_s=end,
                )

            src_resource = src
            src_copy = self.copies.try_get(tensor_id, src_resource)
            if src_copy is None:
                resources = self.copies.resources_for(tensor_id)
                if not resources:
                    raise RuntimePlanError(f"Transfer {inst.name}: no source copy of {tensor_id}")
                src_copy = self.copies.get(tensor_id, resources[0])
                src_resource = src_copy.resource_id

            backend = select_transfer_backend(inst.transfer_backend, destination=dest)
            delay = float(inst.attributes.get("mock_transfer_delay_s", 0.0))
            is_mock = "mock" in dest.lower() or delay > 0 or backend.backend_id == "simulated_device"
            stream = self.streams.copy_stream(dest, delay_s=delay if is_mock else 0.0)

            def _xfer() -> Any:
                out, result = backend.transfer(
                    src_copy.value,
                    source=src_resource,
                    destination=dest,
                    nbytes=inst.nbytes or src_copy.nbytes,
                )
                self.copies.put(
                    tensor_id,
                    dest,
                    out,
                    tier="device" if ("mock" in dest or "gpu" in dest or "cuda" in dest) else "system_ram",
                )
                resources = self.copies.resources_for(tensor_id)
                if len(resources) > 1:
                    self._multi_copy_peaks.append(
                        {
                            "tensor_id": tensor_id,
                            "resources": list(resources),
                            "at": "transfer_complete",
                        }
                    )
                with self._transfer_lock:
                    self._pending_transfers.pop(key, None)
                return result

            fut = stream.submit(_xfer, delay_s=delay if is_mock else 0.0)
            enqueue_end = time.perf_counter()
            with self._transfer_lock:
                self._pending_transfers[key] = fut
            inst.attributes["_async_future"] = fut
            inst.attributes["_enqueue_start_s"] = enqueue_start
            inst.attributes["_enqueue_end_s"] = enqueue_end
            # Transfer instruction completes at enqueue; WaitEvent observes device completion.
            return InstructionEvent(
                name=inst.name,
                opcode=inst.opcode.value,
                resource=str(inst.resource),
                submitted_s=submitted,
                start_s=enqueue_start,
                end_s=enqueue_end,
                nbytes=inst.nbytes or src_copy.nbytes,
                notes="transfer enqueued (async)",
                simulated=is_mock,
                enqueue_start_s=enqueue_start,
                enqueue_end_s=enqueue_end,
            )

        return self._submit_sync(_body)

    def _exec_record(self, inst: PlanInstruction, registry: EventRegistry, submitted: float) -> InstructionEvent:
        start = time.perf_counter()
        # Pair with preceding transfer via attributes.pairs / name convention.
        waits = str(inst.attributes.get("pairs_with_wait") or "")
        transfer_name = inst.depends_on[0] if inst.depends_on else ""
        transfer = self._by_name.get(transfer_name)
        event = StreamEvent(name=inst.name, device=str(inst.resource))
        if transfer is not None:
            fut = transfer.attributes.get("_async_future")
            if isinstance(fut, Future):
                event.bind_future(
                    fut,
                    enqueue_start_s=float(transfer.attributes.get("_enqueue_start_s", start)),
                    enqueue_end_s=float(transfer.attributes.get("_enqueue_end_s", start)),
                )
            else:
                event.record()
        else:
            event.record()
        registry.store(inst.name, event)
        end = time.perf_counter()
        return InstructionEvent(
            name=inst.name,
            opcode=inst.opcode.value,
            resource=str(inst.resource),
            submitted_s=submitted,
            start_s=start,
            end_s=end,
            notes=f"RecordEvent pairs_with={waits}",
            enqueue_start_s=event.enqueue_start_s,
            enqueue_end_s=event.enqueue_end_s,
            complete_s=event.complete_s or end,
        )

    def _exec_wait(self, inst: PlanInstruction, registry: EventRegistry, submitted: float) -> InstructionEvent:
        start = time.perf_counter()
        waits_for = str(inst.attributes.get("waits_for") or (inst.depends_on[0] if inst.depends_on else ""))
        event = registry.get(waits_for)
        # Do not host-sync at Record time; wait here only.
        wait0 = time.perf_counter()
        event.wait()
        wait_s = time.perf_counter() - wait0
        end = time.perf_counter()
        return InstructionEvent(
            name=inst.name,
            opcode=inst.opcode.value,
            resource=str(inst.resource),
            submitted_s=submitted,
            start_s=start,
            end_s=end,
            consumer_wait_s=wait_s,
            exposed_stall_s=wait_s,
            notes=f"WaitEvent on {waits_for}",
            complete_s=event.complete_s or end,
        )

    def _submit_compute(self, inst: PlanInstruction, submitted: float) -> Future[Any]:
        region_id = str(inst.executable_ref or "")
        binding = self.bindings[region_id]
        region = binding.region
        resource = binding.device
        delay = float(inst.attributes.get("mock_compute_delay_s", 0.0))
        if delay <= 0 and "mock" in resource:
            delay = (
                float(binding.compiled.attributes.get("mock_delay_s", 0.05))
                if hasattr(binding.compiled, "attributes")
                else 0.05
            )
        stream = self.streams.compute_stream(
            resource, delay_s=delay if delay > 0 else 0.0, workers=max(4, self.max_inflight)
        )

        # Gather inputs from copies on this resource (or host for CPU).
        args: list[Any] = []
        for name in region.inputs:
            copy = self.copies.try_get(name, resource)
            if copy is None:
                # State may live under env name after Load on this resource.
                if name in self.program.state_bindings:
                    copy = self.copies.try_get(name, resource)
                if copy is None:
                    # Fall back: any copy (same-device plans / host).
                    for rid in self.copies.resources_for(name):
                        copy = self.copies.get(name, rid)
                        break
            if copy is None and name in self.program.state_bindings:
                tensor = self.parameter_store.acquire(name)
                self.copies.put(name, resource, tensor)
                copy = self.copies.get(name, resource)
            if copy is None:
                raise RuntimePlanError(f"Compute {region_id} missing input {name} on {resource}")
            value = copy.value
            from streamcompiler.runtime.activation_spill import is_spilled, reload_spilled

            if is_spilled(value):
                value = reload_spilled(value)
                self.copies.put(name, resource, value)
                self._spill_events.append({"event": "reload", "name": name})
            args.append(value)

        call = self._callables[region_id]

        if self.process_pool is not None and self.fork_registry_id is not None and "mock" not in resource:
            from streamcompiler.runtime.graph_executor import _fork_run_region

            def _detach_arg(value: Any) -> Any:
                if isinstance(value, torch.Tensor):
                    return value.detach()
                if isinstance(value, (tuple, list)):
                    return type(value)(_detach_arg(v) for v in value)
                if isinstance(value, dict):
                    return {k: _detach_arg(v) for k, v in value.items()}
                return value

            pfut = self.process_pool.submit(
                _fork_run_region,
                self.fork_registry_id,
                region_id,
                resource,
                binding.backend_id,
                tuple(_detach_arg(a) for a in args),
            )
            out: Future[Any] = Future()

            def _done_fork(f: Future[Any]) -> None:
                try:
                    region_event, outputs = f.result()
                    for out_name, value in zip(region.outputs, outputs, strict=True):
                        self.copies.put(out_name, resource, value)
                    out.set_result(
                        InstructionEvent(
                            name=inst.name,
                            opcode=inst.opcode.value,
                            resource=resource,
                            submitted_s=submitted,
                            start_s=region_event.start_s,
                            end_s=region_event.end_s,
                            notes=f"Compute {region_id} (process)",
                        )
                    )
                except Exception as exc:
                    out.set_exception(exc)

            pfut.add_done_callback(_done_fork)
            return out

        def _run() -> InstructionEvent:
            start = time.perf_counter()
            if torch.is_inference_mode_enabled():
                result = call(*args)
            else:
                with torch.inference_mode():
                    result = call(*args)
            outputs = coerce_region_result(result)
            if len(outputs) != len(region.outputs):
                raise RuntimePlanError(
                    f"Region {region_id} produced {len(outputs)} values, expected {len(region.outputs)}"
                )
            for out_name, value in zip(region.outputs, outputs, strict=True):
                if self.allocator is not None and isinstance(value, torch.Tensor):
                    slot = self._reuse_assignment.get(out_name)
                    if slot is not None:
                        value = self.allocator.acquire(slot, out_name, value)
                self.copies.put(out_name, resource, value)
            self._maybe_spill_activations(region)
            end = time.perf_counter()
            return InstructionEvent(
                name=inst.name,
                opcode=inst.opcode.value,
                resource=resource,
                submitted_s=submitted,
                start_s=start,
                end_s=end,
                notes=f"Compute {region_id}",
            )

        fut = stream.submit(_run, delay_s=delay if delay > 0 else 0.0)
        # Wrap so Future returns InstructionEvent (stream already ran delay+fn).
        out2: Future[Any] = Future()

        def _done(f: Future[Any]) -> None:
            try:
                out2.set_result(f.result())
            except Exception as exc:
                out2.set_exception(exc)

        fut.add_done_callback(_done)
        return out2

    def _exec_release(self, inst: PlanInstruction, submitted: float) -> InstructionEvent:
        start = time.perf_counter()
        freed = 0
        for tensor_id in inst.inputs:
            # Prefer dropping producer-device copy; attributes may name resource.
            resource = str(inst.attributes.get("release_resource") or inst.resource)
            if self.copies.has(tensor_id, resource):
                freed += self.copies.drop(tensor_id, resource)
            else:
                freed += self.copies.drop(tensor_id, None)
            if tensor_id in self.program.state_bindings or tensor_id in self.program.state_bindings.values():
                self.parameter_store.release((tensor_id,))
        end = time.perf_counter()
        return InstructionEvent(
            name=inst.name,
            opcode=inst.opcode.value,
            resource=str(inst.resource),
            submitted_s=submitted,
            start_s=start,
            end_s=end,
            nbytes=freed,
            notes="schedule Release",
        )

    def _exec_evict(self, inst: PlanInstruction, submitted: float) -> InstructionEvent:
        start = time.perf_counter()
        freed = 0
        for tensor_id in inst.inputs:
            resource = str(inst.destination or inst.resource)
            freed += self.copies.drop(tensor_id, resource)
            if tensor_id in self.program.state_bindings:
                self.parameter_store.release((tensor_id,))
        end = time.perf_counter()
        return InstructionEvent(
            name=inst.name,
            opcode=inst.opcode.value,
            resource=str(inst.resource),
            submitted_s=submitted,
            start_s=start,
            end_s=end,
            nbytes=freed,
            notes="schedule Evict",
        )

    def _maybe_spill_activations(self, region: Any) -> None:
        budget = self.activation_budget_bytes
        if budget is None or not self.allow_activation_spill:
            return
        from streamcompiler.runtime.activation_spill import spill_tensor

        live = 0
        tensors: list[tuple[str, str, Any]] = []
        with self.copies._lock:  # noqa: SLF001
            items = list(self.copies._copies.items())  # noqa: SLF001
        for (tensor_id, resource_id), copy in items:
            if tensor_id in self.program.user_inputs or tensor_id in self.program.state_bindings:
                continue
            if isinstance(copy.value, torch.Tensor):
                nbytes = int(copy.value.numel() * copy.value.element_size())
                live += nbytes
                tensors.append((tensor_id, resource_id, copy.value))
        while live > budget and tensors:
            protected = {str(ref) for kind, ref in self.program.output_refs if kind == "value"}
            candidates = [(t, r, v) for t, r, v in tensors if t not in protected]
            if not candidates:
                break
            tensor_id, resource_id, value = max(
                candidates, key=lambda item: int(item[2].numel() * item[2].element_size())
            )
            spilled = spill_tensor(value)
            self.copies.put(tensor_id, resource_id, spilled, tier="disk")
            live -= spilled.nbytes
            self._spill_events.append({"event": "spill", "name": tensor_id, **spilled.as_dict()})
            tensors = [(t, r, v) for t, r, v in tensors if not (t == tensor_id and r == resource_id)]

    def _collect_outputs(self, host: str) -> list[Any]:
        flat: list[Any] = []
        for kind, ref in self.program.output_refs:
            if kind != "value":
                flat.append(ref)
                continue
            name = str(ref)
            copy = self.copies.try_get(name, host)
            if copy is None:
                resources = self.copies.resources_for(name)
                if not resources:
                    raise RuntimePlanError(f"Missing output {name}")
                copy = self.copies.get(name, resources[0])
            value = copy.value
            from streamcompiler.runtime.activation_spill import is_spilled, reload_spilled

            if is_spilled(value):
                value = reload_spilled(value)
            flat.append(value)
        return flat
