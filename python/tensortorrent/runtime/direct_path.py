"""Low-overhead execution for fully measured resident inference plans.

Single-region plans collapse to one pre-resolved call. A CPU/real-accelerator
plan may also collapse to static dependency waves, but only after compilation
measures that exact dataflow against full fusion. Parameters remain placed
between forwards and are refreshed when their source version changes.

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
    flat inputs, or a parameter tensor already placed on the region's device.
    """

    region_id: str
    device: str
    torch_device: Any
    call: Callable[..., Any]
    # (is_graph_input, index) when True, (False, tensor) when a bound parameter.
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
        """Place ``source`` once on ``torch_device``."""
        value = source.to(torch_device) if torch_device is not None else source
        return cls(
            source=source,
            value=value,
            torch_device=torch_device,
            source_version=int(getattr(source, "_version", 0)),
        )

    def resolve(self) -> Any:
        version = int(getattr(self.source, "_version", 0))
        if version != self.source_version:
            self.value = self.source.to(self.torch_device) if self.torch_device is not None else self.source
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
    # parameters already placed on this region's device.
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
    # Value names that may be dropped after each wave (activation Releases).
    release_after_wave: tuple[tuple[str, ...], ...] = ()
    reason: str = ""

    def refresh_parameters(self) -> None:
        for parameter in self.parameters:
            parameter.resolve()


def _single_compute(schedule: Any) -> Any | None:
    """The one Compute instruction in a static resident schedule.

    Parameter/input transfers and their events are reproduced by pre-placed
    parameters plus input device moves. Stateful storage operations and
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


def _compute_region_dependencies(schedule: Any) -> dict[str, set[str]] | None:
    """Map region_id → upstream Compute regions via the schedule DAG.

    Transfer / event edges between Computes are preserved so waves respect
    stream ordering the producer-only tensor graph would miss.
    """
    by_name = {str(inst.name): inst for inst in schedule.instructions}
    compute_by_region: dict[str, Any] = {}
    for inst in schedule.instructions:
        if inst.opcode != OpCode.COMPUTE:
            continue
        region_id = str(inst.executable_ref or "")
        if not region_id or region_id in compute_by_region:
            return None
        compute_by_region[region_id] = inst

    dependencies: dict[str, set[str]] = {region_id: set() for region_id in compute_by_region}
    for region_id, inst in compute_by_region.items():
        stack = list(inst.depends_on)
        seen: set[str] = set()
        while stack:
            dep_name = str(stack.pop())
            if dep_name in seen:
                continue
            seen.add(dep_name)
            dep = by_name.get(dep_name)
            if dep is None:
                continue
            if dep.opcode == OpCode.COMPUTE:
                upstream = str(dep.executable_ref or "")
                if upstream and upstream != region_id:
                    dependencies[region_id].add(upstream)
                continue
            stack.extend(str(name) for name in dep.depends_on)
    return dependencies


def _seed_schedule_device_param_cache(
    schedule_executor: Any,
    *,
    name: str,
    resource: str,
    parameter: DirectParameter,
) -> None:
    """Share dataflow-placed GPU copies with the schedule fallback path."""
    if parameter.torch_device is None or "mock" in resource.lower():
        return
    from tensortorrent.runtime.copies import describe_tensor
    from tensortorrent.runtime.handles import _tensor_view_meta

    signature = (id(parameter.source), int(parameter.source_version))
    value = parameter.value
    copy_meta = describe_tensor(value, name, resource)
    view_meta = _tensor_view_meta(value)
    key = (name, resource)
    with schedule_executor._persistent_param_lock:
        schedule_executor._persistent_device_param_cache[key] = (
            signature,
            value,
            int(copy_meta.nbytes),
            copy_meta,
            view_meta,
        )


def _build_dataflow_direct_plan(executor: Any, schedule: Any, program: Any) -> DataflowDirectPlan | None:
    """Build a static resident multi-region DAG fast path.

    This is the multi-region form of :class:`DirectPlan`: same compiled region
    callables and placement, but dependencies become precomputed waves instead
    of replaying generic scheduler/residency machinery every forward.
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
    if any(kind != "value" for kind, _ in program.output_refs):
        return None
    region_bindings = [schedule_executor.bindings.get(str(region.region_id)) for region in program.regions]
    backend_ids = {str(getattr(binding, "backend_id", "")) for binding in region_bindings if binding is not None}
    if "cpu" not in backend_ids or not backend_ids.intersection({"cuda", "rocm"}):
        return None

    schedule_deps = _compute_region_dependencies(schedule)
    if schedule_deps is None:
        return None

    state = program.state_tensors()
    user_inputs = tuple(program.user_inputs)
    known_values = set(user_inputs)
    producers: dict[str, str] = {}
    regions: dict[str, DirectRegion] = {}
    parameters: list[DirectParameter] = []
    placed: dict[tuple[int, str], DirectParameter] = {}

    for region in program.regions:
        region_id = str(region.region_id)
        binding = schedule_executor.bindings.get(region_id)
        call = schedule_executor._callables.get(region_id)
        if binding is None or call is None or "mock" in str(binding.device).lower():
            return None
        if region_id not in schedule_deps:
            return None
        torch_device = _resolve_torch_device(binding)
        device_key = str(torch_device) if torch_device is not None else ""
        resource = str(binding.device)
        arg_plan: list[tuple[bool, Any]] = []
        param_bytes = 0
        for name in region.inputs:
            tensor = state.get(name)
            if tensor is not None:
                cache_key = (id(tensor), device_key)
                parameter = placed.get(cache_key)
                if parameter is None:
                    try:
                        parameter = DirectParameter.place(tensor, torch_device)
                    except Exception:  # noqa: BLE001 - placement failure -> scheduler
                        return None
                    placed[cache_key] = parameter
                    parameters.append(parameter)
                    param_bytes += int(parameter.value.numel() * parameter.value.element_size())
                    _seed_schedule_device_param_cache(
                        schedule_executor, name=str(name), resource=resource, parameter=parameter
                    )
                arg_plan.append((False, parameter))
                continue
            if name not in known_values and name not in producers:
                return None
            arg_plan.append((True, str(name)))
        direct_region = DirectRegion(
            region_id=region_id,
            device=resource,
            torch_device=torch_device,
            call=call,
            arg_plan=tuple(arg_plan),
            output_names=tuple(region.outputs),
            param_bytes=param_bytes,
        )
        regions[region_id] = direct_region
        for output in region.outputs:
            producers[str(output)] = region_id
            known_values.add(str(output))

    if len(regions) < 2:
        return None

    # Keep only deps among regions we built; reject unknown upstream Computes.
    dependencies: dict[str, set[str]] = {}
    for region_id, deps in schedule_deps.items():
        if region_id not in regions:
            continue
        if any(dep not in regions for dep in deps):
            return None
        dependencies[region_id] = set(deps)

    waves: list[tuple[DirectRegion, ...]] = []
    remaining = set(regions)
    completed: set[str] = set()
    order = [str(region.region_id) for region in program.regions]
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

    last_use: dict[str, int] = {name: -1 for name in user_inputs}
    for wave_idx, wave in enumerate(waves):
        for region in wave:
            for is_value, slot in region.arg_plan:
                if is_value:
                    last_use[str(slot)] = wave_idx
            for output in region.output_names:
                last_use[str(output)] = wave_idx
    for name in wanted:
        last_use[name] = len(waves)  # live through the final return
    release_after: list[tuple[str, ...]] = []
    for wave_idx in range(len(waves)):
        release_after.append(
            tuple(sorted(name for name, use_wave in last_use.items() if use_wave == wave_idx and name not in wanted))
        )

    return DataflowDirectPlan(
        waves=tuple(waves),
        user_inputs=user_inputs,
        output_refs=tuple(program.output_refs),
        parameters=tuple(parameters),
        param_bytes=sum(region.param_bytes for region in regions.values()),
        release_after_wave=tuple(release_after),
        reason=f"{len(regions)} resident regions in {len(waves)} static dependency waves",
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
