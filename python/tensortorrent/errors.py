"""Public exception hierarchy.

Prefer explicit failures with actionable messages over silent wrong answers.
"""

from __future__ import annotations


class TensorTorrentError(Exception):
    """Base for package errors."""


class UnsupportedFeatureError(TensorTorrentError):
    """Feature not available in this build / milestone."""


class GraphCaptureError(TensorTorrentError):
    """torch.export or IR lowering failed."""


class BackendError(TensorTorrentError):
    """Execution or communication backend failed."""


class PlanningError(TensorTorrentError):
    """No feasible plan."""


class MemoryCapacityError(PlanningError):
    """Plan exceeds measured memory capacities."""


class SpecializationError(TensorTorrentError):
    """Machine specialization failed or needs regenerate."""


class StorageError(TensorTorrentError):
    """Packed model storage I/O failed."""


class RuntimePlanError(TensorTorrentError):
    """Execution plan invalid at runtime."""


class ExecutionCancelled(RuntimePlanError):
    """In-flight ``run`` aborted via cancel."""


class ConfigurationError(TensorTorrentError):
    """CompileConfig or environment invalid / unsupported."""


class DiskSpaceError(StorageError):
    """Not enough disk for a required write."""

    def __init__(self, path: str | object, needed: int, free: int) -> None:
        self.path = str(path)
        self.needed = needed
        self.free = free
        super().__init__(f"Insufficient disk space at {self.path!r}: need {needed} bytes, have {free} bytes")
