"""Explicit tensor transfer backends.

Transfers are plan instructions, not hidden ``tensor.to(device)`` calls inside
compute. CPU hosts implement real disk→RAM and host memcpy paths. When a
destination accelerator is available, ``TorchDeviceTransfer`` performs a real
``Tensor.to``; otherwise device destinations stay on ``SimulatedDeviceTransfer``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol

import torch

from streamcompiler.errors import RuntimePlanError


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


class TorchDeviceTransfer:
    """Real host↔device / device↔device copy via ``Tensor.to``.

    Uses the torch device implied by the schedule destination id
    (``cuda_gpu_0`` → ``cuda:0``, ``cpu_numa_*`` → ``cpu``). When the target
    device is unavailable this backend is not selected.
    """

    backend_id = "torch_device_copy"

    def transfer(
        self,
        value: Any,
        *,
        source: str,
        destination: str,
        nbytes: int,
    ) -> tuple[Any, TransferResult]:
        if not isinstance(value, torch.Tensor):
            raise RuntimePlanError(f"TorchDeviceTransfer needs a tensor, got {type(value)!r}")
        target = torch_device_for_resource(destination)
        start = time.perf_counter()
        # Async when supported; caller may wait via schedule Record/Wait events.
        out = value.to(target, non_blocking=True)
        elapsed = time.perf_counter() - start
        actual = int(out.numel() * out.element_size())
        return out, TransferResult(
            nbytes=actual,
            duration_s=elapsed,
            backend=self.backend_id,
            simulated=False,
            notes=f"torch.to({target}, non_blocking=True) {source}->{destination}",
        )


def torch_device_for_resource(resource: str) -> torch.device:
    """Resolve a schedule resource id through the owning backend."""
    from streamcompiler.backends import backend_by_id, backend_id_for_resource

    backend_id = backend_id_for_resource(resource)
    backend = backend_by_id(backend_id)
    if backend is None:
        raise RuntimePlanError(f"No backend owns resource {resource!r}")
    device = backend.resource_to_torch_device(resource)
    if not isinstance(device, torch.device):
        device = torch.device(device)
    # Availability checks stay backend-specific for accelerators.
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimePlanError(f"CUDA required for device transfer to {resource!r}")
    if device.type == "mps" and (not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available()):
        raise RuntimePlanError(f"MPS required for device transfer to {resource!r}")
    if device.type == "xpu" and (not hasattr(torch, "xpu") or not torch.xpu.is_available()):
        raise RuntimePlanError(f"XPU required for device transfer to {resource!r}")
    return device


# Resource label → torch.device mapping lives on ExecutionBackend.
# Keep torch_device_for_resource for host-side transfer helpers only.


class SimulatedDeviceTransfer:
    """Analytic stand-in for CUDA/ROCm/P2P transfers. Never claims hardware validation."""

    backend_id = "simulated_device"

    def __init__(self, *, bytes_per_s: float = 8e9, latency_s: float = 5e-6, wall_sleep: bool = False) -> None:
        self.bytes_per_s = bytes_per_s
        self.latency_s = latency_s
        self.wall_sleep = wall_sleep

    def transfer(
        self,
        value: Any,
        *,
        source: str,
        destination: str,
        nbytes: int,
    ) -> tuple[Any, TransferResult]:
        # Keep the host tensor as the logical value; device residency is simulated.
        from streamcompiler.runtime.virtual_tensor import VirtualDeviceTensor, wrap_virtual

        duration = self.latency_s + (nbytes / self.bytes_per_s if self.bytes_per_s > 0 else 0.0)
        if self.wall_sleep and duration > 0:
            time.sleep(duration)
        dest_mock = "mock" in destination.lower()
        src_mock = "mock" in source.lower()
        if dest_mock and not src_mock:
            out: Any = wrap_virtual(value, destination)
        elif src_mock and not dest_mock:
            out = value.to_host() if isinstance(value, VirtualDeviceTensor) else value
        elif isinstance(value, VirtualDeviceTensor):
            out = wrap_virtual(value.payload, destination)
        else:
            out = value
        actual = nbytes
        if actual <= 0:
            if isinstance(out, VirtualDeviceTensor):
                actual = out.nbytes
            elif isinstance(out, torch.Tensor):
                actual = int(out.numel() * out.element_size())
        return out, TransferResult(
            nbytes=actual,
            duration_s=duration,
            backend=self.backend_id,
            simulated=True,
            notes=(f"simulated device transfer {source}->{destination}; not hardware-validated"),
        )


def device_transfer_available(destination: str) -> bool:
    """True when a real ``TorchDeviceTransfer`` can target ``destination``."""
    from streamcompiler.backends import backend_id_for_resource

    # Mock / simulated destinations never claim a real torch DMA path.
    if backend_id_for_resource(destination) == "mock_accel":
        return False
    try:
        dev = torch_device_for_resource(destination)
    except RuntimePlanError:
        return False
    if dev.type == "cpu":
        return True
    if dev.type == "cuda":
        return torch.cuda.is_available()
    if dev.type == "mps":
        return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    if dev.type == "xpu":
        return bool(hasattr(torch, "xpu") and torch.xpu.is_available())
    return False


def select_transfer_backend(
    kind: str | None, *, disk_loader: Any = None, destination: str | None = None
) -> TransferBackend:
    if kind == "disk_pread":
        return DiskPreadTransfer(disk_loader)
    if kind in {"device_p2p_or_host_staged", "host_device_copy"}:
        if destination and device_transfer_available(destination):
            return TorchDeviceTransfer()
        return SimulatedDeviceTransfer()
    if kind == "simulated_device":
        return SimulatedDeviceTransfer()
    return HostMemcpyTransfer()
