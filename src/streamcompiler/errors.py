"""Public exception hierarchy for StreamCompiler.

Prefer explicit failures with actionable messages over silent incorrect behavior.
"""

from __future__ import annotations


class StreamCompilerError(Exception):
    """Base class for all StreamCompiler errors."""


class UnsupportedFeatureError(StreamCompilerError):
    """Raised when a requested feature is not supported in the current milestone."""


class GraphCaptureError(StreamCompilerError):
    """Raised when torch.export or IR lowering fails."""


class HardwareError(StreamCompilerError):
    """Raised when hardware discovery or benchmarking fails."""


class BackendError(StreamCompilerError):
    """Raised when an execution or communication backend fails."""


class PlanningError(StreamCompilerError):
    """Raised when no feasible execution plan can be found."""


class MemoryCapacityError(PlanningError):
    """Raised when a plan violates measured memory capacities."""


class SpecializationError(StreamCompilerError):
    """Raised when machine specialization fails or must be regenerated."""


class ValidationError(StreamCompilerError):
    """Raised when hardware or numerical validation fails."""


class StorageError(StreamCompilerError):
    """Raised when packed model storage I/O fails."""


class RuntimePlanError(StreamCompilerError):
    """Raised when an execution plan is invalid at runtime."""
