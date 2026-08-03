"""Zero-overhead execution for plans that do not need a scheduler.

A scheduler earns its keep when there is something to schedule: several
regions, several devices, transfers to overlap, parameters to stream. When the
plan is a single region on a single device with every parameter already
resident, there is nothing to schedule and the dispatch machinery is pure cost.

Profiling the single-region case showed roughly 55% of wall time going to
TensorTorrent's own call stack — module → graph executor → schedule executor →
native bridge → native dispatcher → region handler → compute → callable — for
one function call. That is why small models measured ~2.2x slower than eager.

This module recognises that case at build time and reduces it to what eager
does: resolve the arguments, call the region, return the outputs. Parameter
placement is hoisted out of the forward entirely, exactly as ``nn.Module``
holds its parameters on the device between calls.

The eligibility rules are deliberately strict. Anything that needs ordering,
residency bookkeeping, cancellation mid-forward, or autograd falls back to the
full scheduler, which remains the only path with those guarantees.
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
        return [flat_inputs[slot] if is_input else slot for is_input, slot in self.arg_plan]


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


def build_direct_plan(executor: Any) -> DirectPlan | None:
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
            return None

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
            if torch_device is not None:
                try:
                    tensor = tensor.to(torch_device)
                except Exception:  # noqa: BLE001 - placement failure -> use scheduler
                    return None
            arg_plan.append((False, tensor))

        outputs = tuple(binding.region.outputs)
        wanted = tuple(ref[1] for ref in program.output_refs)
        if any(name not in outputs for name in wanted):
            return None

        param_bytes = sum(
            int(slot.numel() * slot.element_size())
            for is_input, slot in arg_plan
            if not is_input and hasattr(slot, "numel")
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


def _resolve_torch_device(binding: Any) -> Any | None:
    """Torch device for a binding, or ``None`` to leave tensors where they are."""
    backend = getattr(binding, "backend", None)
    resource = str(getattr(binding, "device", ""))
    if backend is not None and hasattr(backend, "resource_to_torch_device"):
        try:
            return backend.resource_to_torch_device(resource)
        except Exception:  # noqa: BLE001 - fall through to the textual mapping
            pass
    return None
