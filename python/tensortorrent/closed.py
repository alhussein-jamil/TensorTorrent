"""Closed categorical values: ``str, Enum`` or ``Literal`` — never bare ``str``.

Prefer domain enums already defined next to their types (``OpCode``, ``MemoryTier``,
``Objective``, …). This module holds cross-cutting and leftover closed sets.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal


class NumericalMode(str, Enum):
    EXACT = "exact"
    QUANTIZED = "quantized"


class ProfileLevel(str, Enum):
    COARSE = "coarse"
    COMPETITIVE = "competitive"
    FULL = "full"


class InstructionKind(str, Enum):
    """Schedule instruction ``attributes["kind"]`` values."""

    PARAMETER_HOST_TO_DEVICE = "parameter_host_to_device"
    PARAMETER_PREFETCH = "parameter_prefetch"
    PARAMETER_MATERIALIZE = "parameter_materialize"
    PARAMETER_EVICT = "parameter_evict"
    ACTIVATION_SPILL = "activation_spill"
    ACTIVATION_RELOAD = "activation_reload"
    ACTIVATION_RELOAD_TRANSFER = "activation_reload_transfer"
    ACTIVATION = "activation"


class CopyOwnership(str, Enum):
    RUNTIME = "runtime"
    ACTIVATION = "activation"
    TRANSFER = "transfer"
    PARAMETER = "parameter"
    INPUT = "input"


class TensorKind(str, Enum):
    """IR ``TensorMeta.kind``."""

    PARAMETER = "parameter"
    ACTIVATION = "activation"
    WORKSPACE = "workspace"


class ValueKind(str, Enum):
    """Region ``ValueSpec.kind``."""

    INPUT = "input"
    PARAMETER = "parameter"
    BUFFER = "buffer"
    CONSTANT = "constant"
    ACTIVATION = "activation"


class ResidencyKind(str, Enum):
    """``ResidencyRequirement.kind``."""

    PARAMETER = "parameter"
    ACTIVATION = "activation"
    INPUT = "input"


class TransferKind(str, Enum):
    P2P = "p2p"
    HOST_STAGED = "host_staged"
    DMA = "dma"
    SHARED = "shared"
    UNSUPPORTED = "unsupported"


class ParameterStoreKind(str, Enum):
    RESIDENT = "resident"
    STREAMING = "streaming"
    EAGER_FUSED = "eager_fused"


class DeviceSelection(str, Enum):
    """Public ``devices=`` presets (not open device instance names)."""

    AUTO = "auto"
    ALL = "all"
    EMPTY = ""
    CPU = "cpu"
    GPU = "gpu"
    CUDA = "cuda"
    ACCELERATOR = "accelerator"


class CompressionKind(str, Enum):
    NONE = "none"
    INT8_AFFINE = "int8_affine"


class BudgetSourceKind(str, Enum):
    EXPLICIT = "explicit"
    CGROUP_V2 = "cgroup_v2"
    CGROUP_V1 = "cgroup_v1"
    OS_AVAILABLE = "os_available"
    TOTAL_FALLBACK = "total_fallback"


class RequestStatus(str, Enum):
    RUNNING = "running"
    OK = "ok"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RequestOutcome(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class HealthStatus(str, Enum):
    OK = "ok"
    STOPPING = "stopping"


class ProfileKind(str, Enum):
    REGION = "region"
    TRANSFER = "transfer"
    OVERLAP = "overlap"
    MEMORY = "memory"


class OutputRefKind(str, Enum):
    VALUE = "value"
    CONSTANT = "constant"


class SimulationStatus(str, Enum):
    """Python mirror of Rust ``SimulationOutcome`` status tags."""

    VALID = "valid"
    INFEASIBLE_MEMORY = "infeasible_memory"
    INVALID_RESIDENCY = "invalid_residency"
    INVALID_EVENT = "invalid_event"
    UNSUPPORTED = "unsupported"


class TensorLayout(str, Enum):
    CONTIGUOUS = "contiguous"
    STRIDED = "strided"


class StartMethod(str, Enum):
    SPAWN = "spawn"
    FORK = "fork"


class LogFormat(str, Enum):
    TEXT = "text"
    JSON = "json"


def closed_str(value: object) -> str:
    """Serialize closed categorical for JSON/attrs.

    On Py3.12+, ``str(SomeEnum.MEMBER)`` is ``\"SomeEnum.MEMBER\"``, not the
    wire value — always prefer ``.value`` for ``str, Enum`` members.
    """
    if isinstance(value, Enum):
        raw = value.value
        return raw if isinstance(raw, str) else str(raw)
    return str(value)


# Aliases kept for call sites that prefer Literal narrowing without Enum members.
NumericalModeStr = Literal["exact", "quantized"]
ProfileLevelStr = Literal["coarse", "competitive", "full"]
InstructionKindStr = Literal[
    "parameter_host_to_device",
    "parameter_prefetch",
    "parameter_materialize",
    "parameter_evict",
    "activation_spill",
    "activation_reload",
    "activation_reload_transfer",
    "activation",
]
CopyOwnershipStr = Literal["runtime", "activation", "transfer", "parameter", "input"]
TensorKindStr = Literal["parameter", "activation", "workspace"]
ValueKindStr = Literal["input", "parameter", "buffer", "constant", "activation"]
ResidencyKindStr = Literal["parameter", "activation", "input"]
TransferKindStr = Literal["p2p", "host_staged", "dma", "shared", "unsupported"]
ParameterStoreKindStr = Literal["resident", "streaming", "eager_fused"]
DeviceSelectionStr = Literal["auto", "all", "", "cpu", "gpu", "cuda", "accelerator"]
CompressionKindStr = Literal["none", "int8_affine"]
BudgetSourceKindStr = Literal["explicit", "cgroup_v2", "cgroup_v1", "os_available", "total_fallback"]
RequestStatusStr = Literal["running", "ok", "failed", "cancelled"]
RequestOutcomeStr = Literal["success", "failed", "cancelled", "timeout"]
HealthStatusStr = Literal["ok", "stopping"]
ProfileKindStr = Literal["region", "transfer", "overlap", "memory"]
OutputRefKindStr = Literal["value", "constant"]
SimulationStatusStr = Literal[
    "valid",
    "infeasible_memory",
    "invalid_residency",
    "invalid_event",
    "unsupported",
]
TensorLayoutStr = Literal["contiguous", "strided"]
StartMethodStr = Literal["spawn", "fork"]
LogFormatStr = Literal["text", "json"]
