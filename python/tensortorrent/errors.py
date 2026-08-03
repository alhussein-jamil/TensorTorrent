"""Public exception hierarchy for TensorTorrent.

Prefer explicit failures with actionable messages over silent incorrect behavior.
"""

from __future__ import annotations


class TensorTorrentError(Exception):
    """Base class for all TensorTorrent errors."""


class UnsupportedFeatureError(TensorTorrentError):
    """Raised when a requested feature is not supported in the current milestone."""


class GraphCaptureError(TensorTorrentError):
    """Raised when torch.export or IR lowering fails."""


class HardwareError(TensorTorrentError):
    """Raised when hardware discovery or benchmarking fails."""


class BackendError(TensorTorrentError):
    """Raised when an execution or communication backend fails."""


class PlanningError(TensorTorrentError):
    """Raised when no feasible execution plan can be found."""


class MemoryCapacityError(PlanningError):
    """Raised when a plan violates measured memory capacities."""


class SpecializationError(TensorTorrentError):
    """Raised when machine specialization fails or must be regenerated."""


class ValidationError(TensorTorrentError):
    """Raised when hardware or numerical validation fails."""


class StorageError(TensorTorrentError):
    """Raised when packed model storage I/O fails."""


class RuntimePlanError(TensorTorrentError):
    """Raised when an execution plan is invalid at runtime."""


class ExecutionCancelled(RuntimePlanError):
    """Raised when ``GraphExecutor.request_cancel`` aborts an in-flight ``run``."""


class ConfigurationError(TensorTorrentError):
    """Raised when a CompileConfig or environment is invalid or unsupported."""


class PlatformError(TensorTorrentError):
    """Raised when the host platform cannot satisfy a requirement."""


class DiskSpaceError(StorageError):
    """Raised when there is insufficient disk space for a required operation.

    Attributes:
        path: The filesystem path where the write was attempted.
        needed: Estimated bytes required.
        free: Bytes actually available.
    """

    def __init__(self, path: str | object, needed: int, free: int) -> None:
        self.path = str(path)
        self.needed = needed
        self.free = free
        super().__init__(f"Insufficient disk space at {self.path!r}: need {needed} bytes, have {free} bytes")
