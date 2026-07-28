"""Explicit tensor transfer backends.

Transfers are plan instructions, not hidden ``tensor.to(device)`` calls inside
compute. CPU hosts implement real disk→RAM and host memcpy paths. Device paths
are interfaces for future CUDA/ROCm streams and peer-to-peer copies.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol

import torch

from streamcompiler.errors import RuntimePlanError
from streamcompiler.runtime.schedule import MemoryTier, PlanInstruction
from streamcompiler.runtime.tensor_directory import TensorDirectory


@dataclass
class TransferResult:
    nbytes: int
    duration_s: float
    backend: str
    simulated: bool = False
    notes: str = ""


class TransferBackend(Protocol):
    backend_id: str

    def transfer(
        self,
        value: Any,
        *,
        source: str,
        destination: str,
        nbytes: int,
    ) -> tuple[Any, TransferResult]: ...


class HostMemcpyTransfer:
    """Real host RAM → host RAM copy (explicit, measurable)."""

    backend_id = "host_memcpy"

    def transfer(
        self,
        value: Any,
        *,
        source: str,
        destination: str,
        nbytes: int,
    ) -> tuple[Any, TransferResult]:
        if not isinstance(value, torch.Tensor):
            raise RuntimePlanError(f"host_memcpy requires a tensor, got {type(value).__name__}")
        start = time.perf_counter()
        out = value.detach().contiguous().clone()
        elapsed = time.perf_counter() - start
        actual = int(out.numel() * out.element_size())
        return out, TransferResult(
            nbytes=actual or nbytes,
            duration_s=elapsed,
            backend=self.backend_id,
            simulated=False,
            notes=f"host memcpy {source}->{destination}",
        )


class DiskPreadTransfer:
    """Real disk→RAM materialization via a callable loader (pack pread)."""

    backend_id = "disk_pread"

    def __init__(self, loader: Any) -> None:
        self._loader = loader

    def transfer(
        self,
        value: Any,
        *,
        source: str,
        destination: str,
        nbytes: int,
    ) -> tuple[Any, TransferResult]:
        start = time.perf_counter()
        if callable(self._loader):
            out = self._loader(value)
        elif isinstance(value, torch.Tensor):
            out = value
        else:
            raise RuntimePlanError("disk_pread transfer needs a loader or tensor")
        elapsed = time.perf_counter() - start
        actual = int(out.numel() * out.element_size()) if isinstance(out, torch.Tensor) else nbytes
        return out, TransferResult(
            nbytes=actual,
            duration_s=elapsed,
            backend=self.backend_id,
            simulated=False,
            notes=f"disk pread {source}->{destination}",
        )


class SimulatedDeviceTransfer:
    """Analytic stand-in for CUDA/ROCm/P2P transfers. Never claims hardware validation."""

    backend_id = "simulated_device"

    def __init__(self, *, bytes_per_s: float = 8e9, latency_s: float = 5e-6) -> None:
        self.bytes_per_s = bytes_per_s
        self.latency_s = latency_s

    def transfer(
        self,
        value: Any,
        *,
        source: str,
        destination: str,
        nbytes: int,
    ) -> tuple[Any, TransferResult]:
        # Keep the host tensor as the logical value; device residency is simulated.
        duration = self.latency_s + (nbytes / self.bytes_per_s if self.bytes_per_s > 0 else 0.0)
        return value, TransferResult(
            nbytes=nbytes,
            duration_s=duration,
            backend=self.backend_id,
            simulated=True,
            notes=(
                f"simulated device transfer {source}->{destination}; "
                "not hardware-validated"
            ),
        )


def select_transfer_backend(kind: str | None, *, disk_loader: Any = None) -> TransferBackend:
    if kind == "disk_pread":
        return DiskPreadTransfer(disk_loader)
    if kind in {"device_p2p_or_host_staged", "host_device_copy", "simulated_device"}:
        return SimulatedDeviceTransfer()
    return HostMemcpyTransfer()


def execute_transfer_instruction(
    inst: PlanInstruction,
    value: Any,
    directory: TensorDirectory,
    *,
    disk_loader: Any = None,
) -> tuple[Any, TransferResult]:
    """Run one Transfer/Prefetch/Load instruction and update residency."""
    if inst.opcode.value not in {"Transfer", "Prefetch", "Load"}:
        raise RuntimePlanError(f"Not a transfer instruction: {inst.opcode}")
    tensor_id = inst.inputs[0] if inst.inputs else inst.name
    dest = inst.destination or inst.resource
    src = inst.source or "unknown"
    backend = select_transfer_backend(inst.transfer_backend, disk_loader=disk_loader)

    # Skip duplicate materialization when a valid copy already sits at dest.
    if directory.has_copy_at(tensor_id, dest) and inst.opcode.value != "Prefetch":
        return value, TransferResult(
            nbytes=0,
            duration_s=0.0,
            backend="elided_duplicate",
            simulated=False,
            notes=f"skipped duplicate transfer of {tensor_id} to {dest}",
        )

    directory.begin_transfer(tensor_id)
    out, result = backend.transfer(value, source=src, destination=dest, nbytes=inst.nbytes)
    tier = inst.memory_tier if isinstance(inst.memory_tier, MemoryTier) else MemoryTier.SYSTEM_RAM
    directory.complete_transfer(
        tensor_id,
        location=dest,
        tier=tier,
        nbytes=result.nbytes,
        device=dest,
    )
    return out, result
