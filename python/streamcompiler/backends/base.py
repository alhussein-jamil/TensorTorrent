"""Stable execution-backend contract.

Planner code must query capabilities through this interface. Vendor-specific
conditionals must not be scattered through the planner.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from streamcompiler.ir.graph import HeterogeneousGraph, Instruction
from streamcompiler.ir.resource_graph import ComputeResource, ResourceGraph


@dataclass(frozen=True)
class KernelCandidate:
    """A concrete kernel realization for a graph region on one device."""

    region_id: str
    device: str
    backend_id: str
    kernel_id: str
    dtype: str
    estimated_latency_s: float | None = None
    workspace_bytes: int = 0
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RegionSource:
    """Hardware-independent description of one compilable region.

    ``module`` is the real subgraph to execute (a ``torch.fx.GraphModule``
    produced by region partitioning). Backends must not rebuild the graph; they
    only realize it for their device.
    """

    region_id: str
    module: Any
    input_names: tuple[str, ...] = ()
    output_names: tuple[str, ...] = ()
    aten_ops: tuple[str, ...] = ()
    example_inputs: tuple[Any, ...] | None = None
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class CompiledRegion:
    """A region realized for one device.

    ``executable`` must be callable: ``executable(*inputs) -> tensor | tuple``.
    Backends must never place status dictionaries or other placeholders here.
    """

    region_id: str
    device: str
    backend_id: str
    executable: Any
    dtype: str
    torch_device: str = "cpu"
    attributes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not callable(self.executable):
            raise TypeError(
                f"CompiledRegion.executable for {self.region_id} must be callable, got {type(self.executable).__name__}"
            )


@dataclass
class BenchmarkResult:
    candidate: KernelCandidate
    latency_s: float
    memory_bytes: int
    measured: bool
    notes: str = ""


@dataclass
class TransferCapability:
    source: str
    destination: str
    kind: str  # p2p | host_staged | dma | shared | unsupported
    measured_bytes_per_s: float | None = None
    measured_latency_s: float | None = None
    notes: str = ""


def region_identifier(region: Any) -> str:
    """Return the stable identifier of any region description."""
    for attr in ("region_id", "name"):
        value = getattr(region, attr, None)
        if isinstance(value, str):
            return value
    raise TypeError(f"Cannot determine region id from {type(region).__name__}")


class ExecutionBackend(ABC):
    """Common contract every accelerator/CPU backend must implement."""

    backend_id: str

    @abstractmethod
    def available(self) -> bool:
        """True when runtime libraries for this backend are importable/usable."""

    @abstractmethod
    def discover_devices(self) -> ResourceGraph:
        """Discover compute/memory/link resources owned by this backend."""

    @abstractmethod
    def supported_ops(self, device: ComputeResource) -> tuple[str, ...]: ...

    @abstractmethod
    def supported_dtypes(self, device: ComputeResource) -> tuple[str, ...]: ...

    @abstractmethod
    def enumerate_kernels(
        self, region: Instruction | HeterogeneousGraph, device: ComputeResource
    ) -> list[KernelCandidate]: ...

    @abstractmethod
    def benchmark(self, candidate: KernelCandidate) -> BenchmarkResult: ...

    @abstractmethod
    def compile(self, region: RegionSource, candidate: KernelCandidate) -> CompiledRegion:
        """Realize ``region`` for ``candidate.device`` as a callable executable."""

    @abstractmethod
    def execute(self, executable: CompiledRegion, inputs: Sequence[Any]) -> tuple[Any, ...]:
        """Run a compiled region on real tensors and return its real outputs."""

    @abstractmethod
    def transfer_capabilities(
        self, source: ComputeResource | str, destination: ComputeResource | str
    ) -> TransferCapability: ...

    def resource_to_torch_device(self, resource_id: str) -> Any:
        """Map a schedule resource id owned by this backend to a ``torch.device``.

        Transfers and residency checks must call this instead of guessing from
        string heuristics. CPU-only backends return ``torch.device('cpu')``.
        """
        import torch

        return torch.device("cpu")

    def validate_basic_execution(self, device: ComputeResource) -> tuple[bool, str]:
        """Optional lightweight smoke test used by `streamcompiler doctor`."""
        try:
            ops = self.supported_ops(device)
            dtypes = self.supported_dtypes(device)
            if not ops:
                return False, "no supported ops reported"
            if not dtypes:
                return False, "no supported dtypes reported"
            return True, f"ops={len(ops)} dtypes={len(dtypes)}"
        except Exception as exc:  # noqa: BLE001 - surface backend failure explicitly
            return False, f"validation failed: {exc}"
