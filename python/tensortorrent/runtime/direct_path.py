"""Low-overhead execution for fully measured resident inference plans.

Single-region plans collapse to one pre-resolved call. A CPU/real-accelerator
plan may also collapse to static dependency waves, but only after compilation
measures that exact dataflow against schedule and fused candidates. Parameters
remain placed between forwards and are refreshed when their source version
changes.

Streaming, training, cancellation, simulated devices, and unmeasured
multi-region plans keep the full scheduler and its residency guarantees.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from tensortorrent.ir.graph import OpCode

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable


@dataclass(frozen=True)
class DirectPlan:
    """A pre-resolved single-region call.

    ``args`` is built once: each entry is either an index into the caller's
    flat inputs, or a :class:`DirectParameter` already placed on device.
    """

    region_id: str
    device: str
    torch_device: Any
    call: Callable[..., Any]
    # (is_graph_input, index) when True, (False, DirectParameter) when bound.
    arg_plan: tuple[tuple[bool, Any], ...]
    output_names: tuple[str, ...]
    # Bytes of the bound parameters. Static, so counted once at build time and
    # reused for every report instead of re-walked each forward.
    param_bytes: int = 0
    reason: str = ""

    def build_args(self, flat_inputs: list[Any]) -> list[Any]:
        args: list[Any] = []
        for is_input, slot in self.arg_plan:
            value = flat_inputs[slot] if is_input else slot.resolve() if isinstance(slot, DirectParameter) else slot
            if is_input and hasattr(value, "to") and self.torch_device is not None:
                value = value.to(self.torch_device)
            args.append(value)
        return args


@dataclass
class DirectParameter:
    """Placed parameter copy refreshed after an in-place source update."""

    source: Any
    value: Any
    torch_device: Any
    source_version: int

    @classmethod
    def place(cls, source: Any, torch_device: Any) -> DirectParameter:
        """Hoist ``source`` onto ``torch_device`` (clone when already local)."""
        if torch_device is not None:
            value = source.to(torch_device)
            if value is source:
                value = source.clone()
        else:
            value = source.clone() if hasattr(source, "clone") else source
        return cls(
            source=source,
            value=value,
            torch_device=torch_device,
            source_version=int(getattr(source, "_version", 0)),
        )

    def resolve(self) -> Any:
        version = int(getattr(self.source, "_version", 0))
        if version != self.source_version:
            if self.torch_device is not None:
                value = self.source.to(self.torch_device)
                self.value = value.clone() if value is self.source else value
            else:
                self.value = self.source.clone() if hasattr(self.source, "clone") else self.source
            self.source_version = version
        return self.value


@dataclass(frozen=True)
class DirectRegion:
    """One pre-resolved call inside a resident multi-region dataflow plan."""

    region_id: str
    device: str
    torch_device: Any
    call: Callable[..., Any]
    # True entries resolve a value produced by the graph; False entries are
    # DirectParameter slots already placed on this region's device.
    arg_plan: tuple[tuple[bool, Any], ...]
    output_names: tuple[str, ...]
    param_bytes: int = 0


@dataclass(frozen=True)
class DataflowDirectPlan:
    """Static resident region DAG executed without schedule bookkeeping."""

    waves: tuple[tuple[DirectRegion, ...], ...]
    user_inputs: tuple[str, ...]
    output_refs: tuple[tuple[str, Any], ...]
    parameters: tuple[DirectParameter, ...] = ()
    param_bytes: int = 0
    reason: str = ""

    def refresh_parameters(self) -> None:
        for parameter in self.parameters:
            parameter.resolve()


def _single_compute(schedule: Any) -> Any | None:
    """The one Compute instruction in a static resident schedule.

    Parameter/input transfers and their events are reproduced by pre-placed
    parameters plus input ``to(device)`` calls. Stateful storage operations and
    multiple Computes still require the scheduler.
    """
    compute = None
    allowed_ancillary = {OpCode.TRANSFER, OpCode.RECORD_EVENT, OpCode.WAIT_EVENT, OpCode.RELEASE}
    for inst in schedule.instructions:
        if inst.opcode in allowed_ancillary:
            continue
        if inst.opcode != OpCode.COMPUTE:
            return None
        if compute is not None:
            return None
        compute = inst
    return compute


def build_direct_plan(executor: Any) -> DirectPlan | DataflowDirectPlan | None:
    """Build a direct-call plan, or return ``None`` when the scheduler is needed.

    Returning ``None`` is always safe: it just means the normal path runs.
    """
    try:
        schedule_executor = getattr(executor, "_schedule_executor", None)
        if schedule_executor is None:
            return None
        schedule = getattr(schedule_executor, "schedule", None)
        program = getattr(executor, "program", None)
        if schedule is None or program is None:
            return None

        inst = _single_compute(schedule)
        if inst is None:
            return _build_dataflow_direct_plan(executor, schedule, program)

        region_id = str(inst.executable_ref or "")
        binding = schedule_executor.bindings.get(region_id)
        call = schedule_executor._callables.get(region_id)
        if binding is None or call is None:
            return None

        # Streaming stores materialise parameters per forward; that is exactly
        # the bookkeeping the scheduler exists to drive.
        store = getattr(executor, "parameter_store", None)
        if getattr(store, "kind", None) != "resident":
            return None
        if getattr(store, "needs_prefetch", False):
            return None

        user_inputs = tuple(program.user_inputs)
        input_index = {name: i for i, name in enumerate(user_inputs)}
        state = program.state_tensors()
        direct_values = set(user_inputs) | set(state)
        for transfer in schedule.instructions:
            if transfer.opcode == OpCode.TRANSFER and any(
                str(name) not in direct_values for name in (*transfer.inputs, *transfer.outputs)
            ):
                return None

        device = str(binding.device)
        torch_device = _resolve_torch_device(binding)

        arg_plan: list[tuple[bool, Any]] = []
        for name in binding.region.inputs:
            if name in input_index:
                arg_plan.append((True, input_index[name]))
                continue
            tensor = state.get(name)
            if tensor is None:
                return None  # neither a graph input nor a known parameter
            try:
                parameter = DirectParameter.place(tensor, torch_device)
            except Exception:  # noqa: BLE001 - placement failure -> use scheduler
                return None
            arg_plan.append((False, parameter))

        outputs = tuple(binding.region.outputs)
        wanted = tuple(ref[1] for ref in program.output_refs)
        if any(name not in outputs for name in wanted):
            return None

        param_bytes = sum(
            int(slot.value.numel() * slot.value.element_size())
            for is_input, slot in arg_plan
            if not is_input and isinstance(slot, DirectParameter)
        )

        return DirectPlan(
            region_id=region_id,
            device=device,
            torch_device=torch_device,
            call=call,
            arg_plan=tuple(arg_plan),
            output_names=outputs,
            param_bytes=param_bytes,
            reason=f"single region {region_id} on {device}, parameters resident",
        )
    except Exception:  # noqa: BLE001 - eligibility must never break compilation
        return None


def _transfer_kinds_ok_for_dataflow(schedule: Any) -> bool:
    """True when every Transfer is a hoistable parameter_host_to_device.

    Activation / collective Transfers need schedule events; dataflow cannot
    pretend ``.to`` reproduces them.
    """
    for inst in schedule.instructions:
        if inst.opcode != OpCode.TRANSFER:
            continue
        if str(inst.attributes.get("kind") or "") != "parameter_host_to_device":
            return False
    return True


def _compute_region_predecessors(schedule: Any) -> dict[str, set[str]]:
    """Map region_id → upstream region_ids via schedule Compute depends_on.

    Walks through Transfer / RecordEvent / WaitEvent / Release edges so waves
    match the ExecutableSchedule critical path, not IR region declaration order.
    """
    by_name = {inst.name: inst for inst in schedule.instructions}
    computes = [inst for inst in schedule.instructions if inst.opcode == OpCode.COMPUTE]
    region_of = {inst.name: str(inst.executable_ref or "") for inst in computes}
    preds: dict[str, set[str]] = {region_of[inst.name]: set() for inst in computes if region_of[inst.name]}

    for inst in computes:
        region_id = region_of[inst.name]
        if not region_id:
            continue
        stack = list(inst.depends_on)
        seen: set[str] = set()
        while stack:
            dep = stack.pop()
            if dep in seen:
                continue
            seen.add(dep)
            other = by_name.get(dep)
            if other is None:
                continue
            if other.opcode == OpCode.COMPUTE:
                upstream = region_of.get(other.name) or str(other.executable_ref or "")
                if upstream and upstream != region_id:
                    preds[region_id].add(upstream)
            else:
                stack.extend(other.depends_on)
    return preds


def _build_dataflow_direct_plan(executor: Any, schedule: Any, program: Any) -> DataflowDirectPlan | None:
    """Build a static resident multi-region DAG fast path.

    Waves come from schedule Compute dependency closure. Transfers must be
    hoistable parameter copies only; Load/Evict/collective Transfers reject.
    """
    schedule_executor = executor._schedule_executor
    store = getattr(executor, "parameter_store", None)
    if (
        not bool(getattr(executor, "_dataflow_direct_path_enabled", False))
        or getattr(store, "kind", None) != "resident"
        or getattr(store, "needs_prefetch", False)
        or int(getattr(executor, "max_workers", 1)) <= 1
    ):
        return None

    allowed = {
        OpCode.COMPUTE,
        OpCode.TRANSFER,
        OpCode.RECORD_EVENT,
        OpCode.WAIT_EVENT,
        OpCode.RELEASE,
    }
    if any(inst.opcode not in allowed for inst in schedule.instructions):
        return None
    if not _transfer_kinds_ok_for_dataflow(schedule):
        return None
    if any(kind != "value" for kind, _ in program.output_refs):
        return None
    region_bindings = [schedule_executor.bindings.get(str(region.region_id)) for region in program.regions]
    backend_ids = {str(getattr(binding, "backend_id", "")) for binding in region_bindings if binding is not None}
    if "cpu" not in backend_ids or not backend_ids.intersection({"cuda", "rocm"}):
        return None

    dependencies = _compute_region_predecessors(schedule)
    state = program.state_tensors()
    user_inputs = tuple(program.user_inputs)
    known_values = set(user_inputs)
    producers: dict[str, str] = {}
    regions: dict[str, DirectRegion] = {}
    parameters: list[DirectParameter] = []

    compute_order = [
        str(inst.executable_ref or "")
        for inst in schedule.instructions
        if inst.opcode == OpCode.COMPUTE and str(inst.executable_ref or "")
    ]
    for region_id in compute_order:
        binding = schedule_executor.bindings.get(region_id)
        call = schedule_executor._callables.get(region_id)
        region = next((r for r in program.regions if str(r.region_id) == region_id), None)
        if binding is None or call is None or region is None or "mock" in str(binding.device).lower():
            return None
        torch_device = _resolve_torch_device(binding)
        arg_plan: list[tuple[bool, Any]] = []
        param_bytes = 0
        for name in region.inputs:
            tensor = state.get(name)
            if tensor is not None:
                try:
                    parameter = DirectParameter.place(tensor, torch_device)
                except Exception:  # noqa: BLE001 - placement failure -> schedule
                    return None
                parameters.append(parameter)
                arg_plan.append((False, parameter))
                param_bytes += int(parameter.value.numel() * parameter.value.element_size())
                continue
            if name not in known_values and name not in producers:
                return None
            arg_plan.append((True, str(name)))
        direct_region = DirectRegion(
            region_id=region_id,
            device=str(binding.device),
            torch_device=torch_device,
            call=call,
            arg_plan=tuple(arg_plan),
            output_names=tuple(region.outputs),
            param_bytes=param_bytes,
        )
        regions[region_id] = direct_region
        dependencies.setdefault(region_id, set())
        for output in region.outputs:
            producers[str(output)] = region_id
            known_values.add(str(output))

    if len(regions) < 2:
        return None
    for region_id, deps in list(dependencies.items()):
        dependencies[region_id] = {d for d in deps if d in regions}

    waves: list[tuple[DirectRegion, ...]] = []
    remaining = set(regions)
    completed: set[str] = set()
    order = list(compute_order)
    while remaining:
        ready = [region_id for region_id in order if region_id in remaining and dependencies[region_id] <= completed]
        if not ready:
            return None
        waves.append(tuple(regions[region_id] for region_id in ready))
        completed.update(ready)
        remaining.difference_update(ready)

    if max(len(wave) for wave in waves) < 2:
        return None
    wanted = {str(ref) for kind, ref in program.output_refs if kind == "value"}
    if not wanted <= set(producers):
        return None
    return DataflowDirectPlan(
        waves=tuple(waves),
        user_inputs=user_inputs,
        output_refs=tuple(program.output_refs),
        parameters=tuple(parameters),
        param_bytes=sum(region.param_bytes for region in regions.values()),
        reason=f"{len(regions)} resident regions in {len(waves)} schedule-derived waves",
    )


def _resolve_torch_device(binding: Any) -> Any | None:
    """Torch device for a binding, or ``None`` to leave tensors where they are."""
    backend = getattr(binding, "backend", None)
    resource = str(getattr(binding, "device", ""))
    if backend is None:
        from tensortorrent.backends import backend_by_id

        backend = backend_by_id(str(getattr(binding, "backend_id", "") or ""))
    if backend is not None and hasattr(backend, "resource_to_torch_device"):
        try:
            return backend.resource_to_torch_device(resource)
        except Exception:  # noqa: BLE001 - fall through to the textual mapping
            pass
    return None
