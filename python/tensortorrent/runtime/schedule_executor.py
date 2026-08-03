"""Instruction-DAG executor: ExecutableSchedule is the exclusive runtime program.

Every Prefetch / Load / Transfer / RecordEvent / WaitEvent / Compute / Evict /
Release op is dispatched when its ``depends_on`` instructions have completed.
Independent instructions may overlap; compute order need not match region order.
"""

from __future__ import annotations

import contextlib
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import torch

from tensortorrent.backends.torch_device import coerce_region_result
from tensortorrent.compile.regions import RegionBinding, RegionProgram
from tensortorrent.errors import RuntimePlanError
from tensortorrent.ir.graph import OpCode
from tensortorrent.runtime.copies import CopyStore
from tensortorrent.runtime.execution_context import ExecutionContext
from tensortorrent.runtime.schedule import ExecutableSchedule, PlanInstruction
from tensortorrent.runtime.tensor_store import ParameterStore


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
    region_id: str | None = None

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)


def max_concurrency_from_intervals(intervals: list[tuple[float, float]]) -> int:
    """Peak concurrency via sweep over half-open ``[start, end)`` intervals."""
    points: list[tuple[float, int]] = []
    for start, end in intervals:
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            continue
        points.append((start, 1))
        points.append((end, -1))
    # Ends (-1) before starts (+1) at the same timestamp.
    points.sort(key=lambda p: (p[0], p[1]))
    cur = peak = 0
    for _, delta in points:
        cur += delta
        if cur > peak:
            peak = cur
    return peak


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
    activation_bytes_written: int = 0
    activation_bytes_read: int = 0
    spill_latency_s: float = 0.0
    reload_latency_s: float = 0.0
    allocation_peak_bytes: int = 0
    spill_events: list[dict[str, Any]] = field(default_factory=list)

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
            "activation_bytes_written": self.activation_bytes_written,
            "activation_bytes_read": self.activation_bytes_read,
            "spill_latency_s": self.spill_latency_s,
            "reload_latency_s": self.reload_latency_s,
            "allocation_peak_bytes": self.allocation_peak_bytes,
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
        max_inflight: int = 8,
        max_workers: int = 1,
        process_pool: Any | None = None,
        fork_registry_id: int | None = None,
        callables: dict[str, Any] | None = None,
        allocator: Any | None = None,
        activation_budget_bytes: int | None = None,
        spill_events: list[dict[str, Any]] | None = None,
        reuse_assignment: dict[str, int] | None = None,
        machine: Any | None = None,
        device_workers: Any | None = None,
    ) -> None:
        from tensortorrent.runtime.schedule import ScheduleValidationError, ensure_explicit_streams, validate_schedule

        schedule = ensure_explicit_streams(schedule)
        violations = validate_schedule(schedule)
        if violations:
            raise RuntimePlanError(
                f"ExecutableSchedule {schedule.graph_name!r} failed validation: {violations}"
            ) from ScheduleValidationError(str(violations))
        self.program = program
        self.bindings = bindings
        self.schedule = schedule
        self.parameter_store = parameter_store
        self.max_inflight = max(1, int(max_inflight))
        self.max_workers = max(1, int(max_workers))
        self.process_pool = process_pool
        self.fork_registry_id = fork_registry_id
        self.device_workers = device_workers
        self.allocator = allocator
        self.activation_budget_bytes = activation_budget_bytes
        self._spill_events = spill_events if spill_events is not None else []
        self._reuse_assignment = dict(reuse_assignment or {})
        self.machine = machine
        # Last-run residency snapshot only; live copies live on ExecutionContext.
        self.copies = CopyStore()
        self._by_name = {i.name: i for i in schedule.instructions}
        if callables is not None:
            self._callables = callables
        else:
            self._callables = {
                rid: getattr(binding.compiled, "executable", binding.compiled) for rid, binding in bindings.items()
            }
        from tensortorrent.runtime.inflight import InFlightGate

        self._run_gate = InFlightGate()
        self._cancel = False
        self._cancel_lock = threading.Lock()
        self._active_cancels: list[Any] = []
        self._closed = False
        # Region-wave pool for concurrent Computes.
        self._region_pool: ThreadPoolExecutor | None = None
        self._native_artifact: Any | None = None
        self._install_native_artifact(schedule)

    def _ensure_region_pool(self, workers: int) -> ThreadPoolExecutor:
        """Thread pool for independent Compute waves on the native path."""
        n = max(1, int(workers))
        if self._region_pool is None:
            self._region_pool = ThreadPoolExecutor(
                max_workers=n,
                thread_name_prefix="tt-region",
            )
            return self._region_pool
        if int(getattr(self._region_pool, "_max_workers", n)) < n:
            self._region_pool.shutdown(wait=False, cancel_futures=True)
            self._region_pool = ThreadPoolExecutor(
                max_workers=n,
                thread_name_prefix="tt-region",
            )
        return self._region_pool

    def _install_native_artifact(self, schedule: ExecutableSchedule) -> None:
        from tensortorrent.native import require_native

        native = require_native()
        self._native_artifact = native.NativeCompiledArtifact.from_schedule(schedule)

    def close(self) -> None:
        if self._closed:
            return
        self.request_cancel()
        self._run_gate.mark_closed_and_wait()
        if self._closed:
            return
        self._closed = True
        self._cancel = True
        self._persistent_param_cache = None
        if self._region_pool is not None:
            self._region_pool.shutdown(wait=True, cancel_futures=True)
            self._region_pool = None
        self._last_native_ctx = None

    def replace_schedule(self, schedule: ExecutableSchedule) -> None:
        """Install a new immutable schedule (e.g. attribute annotations for tests)."""
        from tensortorrent.runtime.schedule import ScheduleValidationError, ensure_explicit_streams, validate_schedule

        self._run_gate.wait_idle()
        if self._closed:
            raise RuntimePlanError("ScheduleExecutor is closed")
        schedule = ensure_explicit_streams(schedule)
        violations = validate_schedule(schedule)
        if violations:
            raise RuntimePlanError(
                f"ExecutableSchedule {schedule.graph_name!r} failed validation: {violations}"
            ) from ScheduleValidationError(str(violations))
        self.schedule = schedule
        self._by_name = {i.name: i for i in schedule.instructions}
        self._install_native_artifact(schedule)

    def request_cancel(self) -> None:
        with self._cancel_lock:
            self._cancel = True
            tokens = list(self._active_cancels)
        for tok in tokens:
            with contextlib.suppress(Exception):
                tok.cancel()

    def run(
        self,
        flat_inputs: list[Any],
        *,
        cancel_token: Any | None = None,
        enable_grad: bool = False,
    ) -> tuple[list[Any], ScheduleReport]:
        if self._closed:
            raise RuntimePlanError("ScheduleExecutor is closed")
        try:
            self._run_gate.enter()
        except RuntimeError as exc:
            raise RuntimePlanError("ScheduleExecutor is closed") from exc
        try:
            from tensortorrent.runtime.native_bridge import run_schedule_native

            return run_schedule_native(self, flat_inputs, cancel_token=cancel_token, enable_grad=bool(enable_grad))
        finally:
            self._run_gate.leave()

    def _default_host_resource(self) -> str:
        for binding in self.bindings.values():
            if "cpu" in binding.device or "numa" in binding.device:
                return binding.device
        return "cpu"

    def _protected_budget_tensors(self) -> set[str]:
        """Tensors the planner refuses to spill (inputs + graph outputs)."""
        protected: set[str] = set(self.program.user_inputs)
        for kind, ref in getattr(self.program, "output_refs", ()):
            if kind == "value":
                protected.add(str(ref))
        protected.update(getattr(self.program, "user_outputs", ()) or ())
        return protected

    def _pending_spill_tensors(self, completed: set[str]) -> set[str]:
        """Activation tensors still waiting on a scheduled spill Evict."""
        pending: set[str] = set()
        for inst in self.schedule.instructions:
            if inst.name in completed:
                continue
            if inst.opcode == OpCode.EVICT and inst.attributes.get("kind") == "activation_spill":
                pending.update(inst.inputs)
        return pending

    def _assert_activation_budget(self, ctx: ExecutionContext, completed: set[str]) -> None:
        """Fail when durable activation residency exceeds the configured budget.

        Matches planner semantics: disk spills and protected tensors may leave
        live bytes above budget only while a pending spill still covers every
        spillable resident activation. Transient overage between Compute and its
        dependent Evict is allowed; leftover spillable residency is not.
        """
        budget = self.activation_budget_bytes
        live = ctx.copies.activation_live_bytes()
        ctx.note_activation_live(live)
        if budget is None:
            return
        if live <= int(budget):
            return
        protected = self._protected_budget_tensors()
        pending = self._pending_spill_tensors(completed)
        spillable = sorted(
            tid for tid in ctx.copies.activation_tensor_ids() if tid not in protected and tid not in pending
        )
        if not spillable:
            return
        raise RuntimePlanError(f"activation budget {int(budget)} bytes exceeded: live={live} spillable={spillable}")

    # ---- Production Compute (also used by native_bridge region callback) ----

    def _exec_compute(self, inst: PlanInstruction, ctx: ExecutionContext, submitted: float) -> InstructionEvent:
        region_id = str(inst.executable_ref or "")
        binding = self.bindings[region_id]
        region = binding.region
        resource = binding.device
        enable_grad = bool(getattr(ctx, "enable_grad", False))

        from tensortorrent.runtime.activation_spill import is_spilled
        from tensortorrent.runtime.virtual_tensor import unwrap_for_compute

        args: list[Any] = []
        for name in region.inputs:
            copy = None
            value: Any = None
            if ctx.copies.has(name, resource, valid_only=True):
                copy = ctx.copies.require(name, resource)
                value = copy.value
                if ctx.native_residency is not None and not ctx.native_residency.session.has(name, resource):
                    raise RuntimePlanError(
                        f"Compute {region_id} on {resource}: CopyStore has {name!r} but native "
                        f"residency does not (Rust is residency authority)"
                    )
            elif ctx.native_residency is not None and ctx.native_residency.session.has(name, resource):
                # Native Transfer may have registered dest residency before CopyStore.
                value = ctx.native_residency.require_value(name, resource)
                from tensortorrent.runtime.virtual_tensor import VirtualDeviceTensor, wrap_virtual_native

                nctx = getattr(ctx, "native_execution_context", None)
                # Training keeps live host tensors on mock resources (byte wrap detaches).
                if not enable_grad:
                    if "mock" in resource.lower() and not isinstance(value, VirtualDeviceTensor):
                        if nctx is None:
                            raise RuntimePlanError(f"Compute {region_id}: mock wrap requires NativeExecutionContext")
                        value = wrap_virtual_native(value, resource, nctx)
                    elif "mock" not in resource.lower() and isinstance(value, VirtualDeviceTensor):
                        value = value.to_host()
                elif isinstance(value, VirtualDeviceTensor):
                    value = value.payload
                if ctx.copies.has(name, ctx.host_resource):
                    ctx.copies.replicate(
                        name,
                        resource,
                        value,
                        ownership="transfer",
                        source_resource=ctx.host_resource,
                    )
                else:
                    ctx.copies.put(name, resource, value, ownership="transfer")
                # Native already has residency; only bind a freshly wrapped virtual buffer.
                if nctx is not None and isinstance(value, VirtualDeviceTensor) and value.native_buffer_id is not None:
                    nctx.bind_virtual_buffer(name, resource, int(value.native_buffer_id))
                copy = ctx.copies.require(name, resource)
            else:
                raise RuntimePlanError(
                    f"Compute {region_id} on {resource}: required copy of {name!r} missing "
                    f"(schedule must Load/Transfer before Compute; no hidden materialization)"
                )
            if is_spilled(copy.value):
                raise RuntimePlanError(
                    f"Compute {region_id}: {name!r} still spilled on {resource!r}; "
                    f"schedule must emit activation_reload Load before Compute"
                )
            args.append(unwrap_for_compute(copy.value, resource=resource, allow_host_alias=enable_grad))

        call = self._callables[region_id]

        if enable_grad:
            workers = self.device_workers
            if workers is not None and resource in getattr(workers, "device_ids", ()):
                raise RuntimePlanError(
                    f"Compute {region_id}: schedule training cannot use device workers "
                    "(they detach tensors). Compile with in-process execution for training."
                )
            if self.process_pool is not None and self.fork_registry_id is not None and "mock" not in resource:
                raise RuntimePlanError(
                    f"Compute {region_id}: schedule training cannot use process_workers "
                    "(fork detaches tensors). Set process_workers=0 for training."
                )

        workers = self.device_workers
        if not enable_grad and workers is not None and resource in getattr(workers, "device_ids", ()):
            from tensortorrent.runtime.device_workers import run_region_on_device

            region_event, outputs = workers.submit(
                resource,
                run_region_on_device,
                call,
                resource,
                binding.backend_id,
                region_id,
                tuple(_detach_for_worker(a) for a in args),
            ).result()
            for out_name, value in zip(region.outputs, outputs, strict=True):
                ctx.copies.put(out_name, resource, value, ownership="activation")
                ctx.mirror_native_put(out_name, resource, value)
            return InstructionEvent(
                name=inst.name,
                opcode=inst.opcode.value,
                resource=resource,
                submitted_s=submitted,
                start_s=region_event["start_s"],
                end_s=region_event["end_s"],
                notes=f"Compute {region_id} (device-worker)",
                region_id=region_id,
            )

        if (
            not enable_grad
            and self.process_pool is not None
            and self.fork_registry_id is not None
            and "mock" not in resource
        ):
            from tensortorrent.runtime.graph_executor import _fork_run_region

            region_event, outputs = self.process_pool.submit(
                _fork_run_region,
                self.fork_registry_id,
                region_id,
                resource,
                binding.backend_id,
                tuple(_detach_for_worker(a) for a in args),
            ).result()
            for out_name, value in zip(region.outputs, outputs, strict=True):
                ctx.copies.put(out_name, resource, value, ownership="activation")
                ctx.mirror_native_put(out_name, resource, value)
            return InstructionEvent(
                name=inst.name,
                opcode=inst.opcode.value,
                resource=resource,
                submitted_s=submitted,
                start_s=region_event.start_s,
                end_s=region_event.end_s,
                notes=f"Compute {region_id} (process)",
                region_id=region_id,
            )

        start = time.perf_counter()
        if enable_grad or torch.is_inference_mode_enabled():
            result = call(*args)
        else:
            with torch.inference_mode():
                result = call(*args)
        outputs = coerce_region_result(result)
        if len(outputs) != len(region.outputs):
            raise RuntimePlanError(f"Region {region_id} produced {len(outputs)} values, expected {len(region.outputs)}")
        for out_name, value in zip(region.outputs, outputs, strict=True):
            # Buffer reuse overwrites storage; unsafe while autograd holds saved tensors.
            if not enable_grad and self.allocator is not None and isinstance(value, torch.Tensor):
                slot = self._reuse_assignment.get(out_name)
                if slot is not None:
                    value = self.allocator.acquire(slot, out_name, value)
            from tensortorrent.runtime.virtual_tensor import VirtualDeviceTensor, wrap_virtual_native

            nctx = getattr(ctx, "native_execution_context", None)
            # Inference mock path owns a native virtual buffer; training keeps the
            # live activation tensor so backward can see grad_fn.
            if "mock" in resource and not enable_grad:
                if nctx is None:
                    raise RuntimePlanError(f"Compute {region_id}: mock wrap requires NativeExecutionContext")
                value = wrap_virtual_native(value, resource, nctx)
            ctx.copies.put(out_name, resource, value, ownership="activation")
            ctx.mirror_native_put(out_name, resource, value)
            if nctx is not None and isinstance(value, VirtualDeviceTensor) and value.native_buffer_id is not None:
                nctx.bind_virtual_buffer(out_name, resource, int(value.native_buffer_id))
        end = time.perf_counter()
        return InstructionEvent(
            name=inst.name,
            opcode=inst.opcode.value,
            resource=resource,
            submitted_s=submitted,
            start_s=start,
            end_s=end,
            notes=f"Compute {region_id}",
            region_id=region_id,
        )

    def _collect_outputs(self, ctx: ExecutionContext) -> list[Any]:
        host = ctx.host_resource
        flat: list[Any] = []
        for kind, ref in self.program.output_refs:
            if kind != "value":
                flat.append(ref)
                continue
            name = str(ref)
            resources = ctx.copies.resources_for(name)
            if not resources:
                raise RuntimePlanError(f"Missing output {name}")
            # Prefer accelerator-resident copies so CompiledModule matches
            # ``nn.Module`` device semantics (outputs stay on the compute device).
            # Host is the fallback when no device copy exists.
            chosen = next((r for r in resources if _tier_is_device(r)), None)
            if chosen is None:
                chosen = host if host in resources else resources[0]
            copy = ctx.copies.get(name, chosen)
            value = copy.value
            from tensortorrent.runtime.activation_spill import is_spilled
            from tensortorrent.runtime.virtual_tensor import VirtualDeviceTensor

            if is_spilled(value):
                raise RuntimePlanError(f"Output {name!r} still spilled; schedule must reload before collect")
            if isinstance(value, VirtualDeviceTensor):
                value = value.to_host()
            flat.append(value)
        return flat


def _tier_is_device(resource: str) -> bool:
    from tensortorrent.backends import backend_id_for_resource
    from tensortorrent.runtime.resource_names import is_device_resource

    return is_device_resource(resource) or backend_id_for_resource(resource) != "cpu"


def _detach_for_worker(value: Any) -> Any:
    """Detach tensors before process/device-worker submit (breaks autograd by design)."""
    if isinstance(value, torch.Tensor):
        return value.detach()
    if isinstance(value, (tuple, list)):
        return type(value)(_detach_for_worker(v) for v in value)
    if isinstance(value, dict):
        return {k: _detach_for_worker(v) for k, v in value.items()}
    return value
