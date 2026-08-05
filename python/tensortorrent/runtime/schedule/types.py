"""Executable schedule types."""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from tensortorrent.ir.graph import OpCode


class MemoryTier(str, Enum):
    DISK = "disk"
    SYSTEM_RAM = "system_ram"
    PINNED_RAM = "pinned_ram"
    NUMA_RAM = "numa_ram"
    DEVICE = "device"
    """Virtual accelerator or future GPU VRAM — never created by Load alone."""
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class FrozenAttrs:
    """Picklable immutable string-key mapping for instruction attributes."""

    _items: tuple[tuple[str, Any], ...] = ()

    @classmethod
    def from_mapping(cls, attrs: Mapping[str, Any] | None) -> FrozenAttrs:
        if isinstance(attrs, FrozenAttrs):
            return attrs
        items = tuple(sorted(((str(k), v) for k, v in dict(attrs or {}).items()), key=lambda kv: kv[0]))
        return cls(items)

    def get(self, key: str, default: Any = None) -> Any:
        for k, v in self._items:
            if k == key:
                return v
        return default

    def __getitem__(self, key: str) -> Any:
        for k, v in self._items:
            if k == key:
                return v
        raise KeyError(key)

    def __contains__(self, key: object) -> bool:
        return any(k == key for k, _ in self._items)

    def __iter__(self) -> Iterator[str]:
        return (k for k, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def keys(self) -> tuple[str, ...]:
        return tuple(k for k, _ in self._items)

    def values(self) -> tuple[Any, ...]:
        return tuple(v for _, v in self._items)

    def items(self) -> tuple[tuple[str, Any], ...]:
        return self._items

    def as_dict(self) -> dict[str, Any]:
        return dict(self._items)


@dataclass(frozen=True)
class PlanInstruction:
    """One scheduled op the runtime can execute and the simulator can cost.

    Immutable: no futures, tensors, timestamps, or runtime handles may be stored
    in ``attributes``. Per-call state lives in :class:`ExecutionContext`.
    """

    opcode: OpCode
    name: str
    resource: str
    depends_on: tuple[str, ...] = ()
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    nbytes: int = 0
    memory_tier: MemoryTier = MemoryTier.UNKNOWN
    predicted_duration_s: float = 0.0
    executable_ref: str | None = None
    """Region id or compiled-region key when opcode is Compute."""
    source: str | None = None
    destination: str | None = None
    backend_id: str | None = None
    transfer_backend: str | None = None
    sync_required: bool = False
    stream_id: str | None = None
    """Ordered stream on resource (compute / copy / io / sync)."""
    copy_engine_id: str | None = None
    """Copy-engine identity for Transfer / Prefetch / Load."""
    link_id: str | None = None
    """Interconnect identity for Transfer."""
    io_queue_id: str | None = None
    """Disk / pack I/O queue identity for Prefetch / Load."""
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.nbytes, bool) or not isinstance(self.nbytes, int):
            raise TypeError(f"Instruction nbytes must be an integer, got {type(self.nbytes).__name__}")
        if self.nbytes < 0:
            raise ValueError(f"Instruction nbytes must be >= 0, got {self.nbytes}")
        if isinstance(self.predicted_duration_s, bool) or not isinstance(self.predicted_duration_s, (int, float)):
            raise TypeError("Instruction predicted_duration_s must be numeric")
        if not math.isfinite(float(self.predicted_duration_s)) or self.predicted_duration_s < 0:
            raise ValueError("Instruction predicted_duration_s must be finite and >= 0")
        if not isinstance(self.sync_required, bool):
            raise TypeError("Instruction sync_required must be a bool")
        for field_name, values in (
            ("depends_on", self.depends_on),
            ("inputs", self.inputs),
            ("outputs", self.outputs),
        ):
            if isinstance(values, (str, bytes)) or any(not isinstance(value, str) for value in values):
                raise TypeError(f"Instruction {field_name} must contain strings")
        object.__setattr__(self, "depends_on", tuple(self.depends_on))
        object.__setattr__(self, "inputs", tuple(self.inputs))
        object.__setattr__(self, "outputs", tuple(self.outputs))
        object.__setattr__(self, "attributes", FrozenAttrs.from_mapping(self.attributes))

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["opcode"] = self.opcode.value
        payload["memory_tier"] = self.memory_tier.value
        payload["attributes"] = dict(self.attributes)
        return payload


@dataclass(frozen=True)
class ExecutableSchedule:
    """Immutable executable plan: same object for plan explain, sim, and run."""

    graph_name: str
    fingerprint: str
    instructions: tuple[PlanInstruction, ...] = ()
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "instructions", tuple(self.instructions))
        object.__setattr__(self, "notes", tuple(self.notes))

    def compute_ops(self) -> list[PlanInstruction]:
        return [i for i in self.instructions if i.opcode == OpCode.COMPUTE]

    def transfer_ops(self) -> list[PlanInstruction]:
        return [i for i in self.instructions if i.opcode in (OpCode.TRANSFER, OpCode.PREFETCH, OpCode.LOAD)]

    def as_dict(self) -> dict[str, Any]:
        return {
            "graph_name": self.graph_name,
            "fingerprint": self.fingerprint,
            "instructions": [i.as_dict() for i in self.instructions],
            "notes": list(self.notes),
        }


class ScheduleValidationError(ValueError):
    """Raised when an :class:`ExecutableSchedule` violates a structural invariant."""
