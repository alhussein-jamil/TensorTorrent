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
    call: Callable[..., Any]
    # (is_graph_input, index) when True, (False, tensor) when a bound parameter.
    arg_plan: tuple[tuple[bool, Any], ...]
    output_names: tuple[str, ...]
    # Bytes of the bound parameters. Static, so counted once at build time and
    # reused for every report instead of re-walked each forward.
    param_bytes: int = 0
    reason: str = ""

    def build_args(self, flat_inputs: list[Any]) -> list[Any]:
        return [
            flat_inputs[slot] if is_input else slot.resolve() if isinstance(slot, DirectParameter) else slot
            for is_input, slot in self.arg_plan
        ]


@dataclass
class DirectParameter:
    """Placed parameter copy refreshed after an in-place source update."""

    source: Any
    value: Any
    torch_device: Any
    source_version: int

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
    reason: str = ""

    def refresh_parameters(self) -> None:
        for parameter in self.parameters:
            parameter.resolve()


def _single_compute(schedule: Any) -> Any | None:
    """The one Compute instruction, if the schedule contains nothing else.

    Any other instruction — a transfer, a load, an evict, an event — means the
    schedule expresses ordering or residency that the direct path does not
    reproduce, so it is not eligible.
    """
    compute = None
    for inst in schedule.instructions:
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
            # Hoist placement out of the forward: bind the parameter on the
            # region's device once, the way an nn.Module holds its weights.
            source = tensor
            if torch_device is not None:
                try:
                    tensor = tensor.to(torch_device)
                except Exception:  # noqa: BLE001 - placement failure -> use scheduler
                    return None
            arg_plan.append(
                (
                    False,
                    DirectParameter(
                        source=source,
                        value=tensor,
                        torch_device=torch_device,
                        source_version=int(getattr(source, "_version", 0)),
                    ),
                )
            )

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
            call=call,
            arg_plan=tuple(arg_plan),
            output_names=outputs,
            param_bytes=param_bytes,
            reason=f"single region {region_id} on {device}, parameters resident",
        )
    except Exception:  # noqa: BLE001 - eligibility must never break compilation
        return None


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

    state = program.state_tensors()
    user_inputs = tuple(program.user_inputs)
    known_values = set(user_inputs)
    producers: dict[str, str] = {}
    regions: dict[str, DirectRegion] = {}
    dependencies: dict[str, set[str]] = {}
    parameters: list[DirectParameter] = []

    for region in program.regions:
        region_id = str(region.region_id)
        binding = schedule_executor.bindings.get(region_id)
        call = schedule_executor._callables.get(region_id)
        if binding is None or call is None or "mock" in str(binding.device).lower():
            return None
        torch_device = _resolve_torch_device(binding)
        arg_plan: list[tuple[bool, Any]] = []
        deps: set[str] = set()
        param_bytes = 0
        for name in region.inputs:
            tensor = state.get(name)
            if tensor is not None:
                source = tensor
                if torch_device is not None:
                    tensor = tensor.to(torch_device)
                parameter = DirectParameter(
                    source=source,
                    value=tensor,
                    torch_device=torch_device,
                    source_version=int(getattr(source, "_version", 0)),
                )
                parameters.append(parameter)
                arg_plan.append((False, parameter))
                param_bytes += int(tensor.numel() * tensor.element_size())
                continue
            if name not in known_values and name not in producers:
                return None
            arg_plan.append((True, str(name)))
            producer = producers.get(str(name))
            if producer is not None:
                deps.add(producer)
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
        dependencies[region_id] = deps
        for output in region.outputs:
            producers[str(output)] = region_id
            known_values.add(str(output))

    if len(regions) < 2:
        return None

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
    return DataflowDirectPlan(
        waves=tuple(waves),
        user_inputs=user_inputs,
        output_refs=tuple(program.output_refs),
        parameters=tuple(parameters),
        param_bytes=sum(region.param_bytes for region in regions.values()),
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
