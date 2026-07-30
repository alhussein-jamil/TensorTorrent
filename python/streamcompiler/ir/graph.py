"""Heterogeneous IR tensors and instructions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class OpCode(str, Enum):
    COMPUTE = "Compute"
    TRANSFER = "Transfer"
    PREFETCH = "Prefetch"
    EVICT = "Evict"
    LOAD = "Load"
    STORE = "Store"
    COLLECTIVE = "Collective"
    RESHARD = "Reshard"
    BROADCAST = "Broadcast"
    REDUCE = "Reduce"
    RECOMPUTE = "Recompute"
    MATERIALIZE = "Materialize"
    COMPRESS = "Compress"
    DECOMPRESS = "Decompress"
    WAIT_EVENT = "WaitEvent"
    RECORD_EVENT = "RecordEvent"
    RELEASE = "Release"


@dataclass
class TensorMeta:
    tensor_id: str
    shape: tuple[int | str, ...]
    dtype: str
    layout: str = "contiguous"
    size_bytes: int = 0
    alias_group: str | None = None
    mutable: bool = False
    storage_id: str | None = None
    kind: str = "activation"  # parameter | activation | workspace
    home_tier: str | None = None
    current_residency: tuple[str, ...] = ()
    valid_copies: tuple[str, ...] = ()
    allowed_devices: tuple[str, ...] = ()
    precision_alternatives: tuple[str, ...] = ()
    recompute_cost_s: float | None = None
    produced_at: int | None = None
    last_use_at: int | None = None
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class Instruction:
    opcode: OpCode
    name: str
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    device: str | None = None
    memory: str | None = None
    source: str | None = None
    destination: str | None = None
    backend_id: str | None = None
    dtype: str | None = None
    estimated_cost_s: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class HeterogeneousGraph:
    """Hardware-independent heterogeneous IR graph produced by portable compilation."""

    name: str
    tensors: dict[str, TensorMeta] = field(default_factory=dict)
    instructions: list[Instruction] = field(default_factory=list)
    parameters: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    repeated_blocks: tuple[tuple[str, ...], ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_tensor(self, tensor: TensorMeta) -> None:
        self.tensors[tensor.tensor_id] = tensor

    def add_instruction(self, inst: Instruction) -> None:
        self.instructions.append(inst)

    def compute_regions(self) -> list[Instruction]:
        return [i for i in self.instructions if i.opcode == OpCode.COMPUTE]
