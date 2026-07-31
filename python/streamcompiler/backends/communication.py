"""Communication backend contracts and host-staged fallback."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from streamcompiler.errors import UnsupportedFeatureError


@dataclass
class CollectiveCapability:
    backend_id: str
    available: bool
    devices: tuple[str, ...]
    ops: tuple[str, ...]
    notes: str = ""


class CommunicationBackend(ABC):
    backend_id: str

    @abstractmethod
    def available(self) -> bool: ...

    @abstractmethod
    def capabilities(self, devices: tuple[str, ...]) -> CollectiveCapability: ...

    @abstractmethod
    def allreduce(self, tensors: Any, devices: tuple[str, ...]) -> Any: ...


def _pg_allreduce(tensors: Any) -> Any | None:
    """Use an initialized torch.distributed process group when present."""
    import torch

    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return None
    if isinstance(tensors, torch.Tensor):
        handle = tensors.detach().clone()
        torch.distributed.all_reduce(handle)
        return handle
    raise UnsupportedFeatureError("process-group allreduce expects a single tensor")


class HostStagedComm(CommunicationBackend):
    """Portable collective path via host memory (mixed-vendor safe)."""

    backend_id = "host_staged"

    def available(self) -> bool:
        return True

    def capabilities(self, devices: tuple[str, ...]) -> CollectiveCapability:
        return CollectiveCapability(
            backend_id=self.backend_id,
            available=True,
            devices=devices,
            ops=("broadcast", "reduce", "allreduce", "gather", "scatter"),
            notes="Host-memory collectives for mixed-device graphs",
        )

    def allreduce(self, tensors: Any, devices: tuple[str, ...]) -> Any:
        """Sum tensors on the host. Single tensor is a no-op clone."""
        import torch

        _ = devices
        if isinstance(tensors, torch.Tensor):
            return tensors.detach().cpu().clone()
        if isinstance(tensors, (list, tuple)) and tensors:
            out = tensors[0].detach().cpu().clone()
            for tensor in tensors[1:]:
                out += tensor.detach().cpu().to(dtype=out.dtype, device=out.device)
            return out
        raise UnsupportedFeatureError("host_staged allreduce requires a tensor or a non-empty sequence of tensors")


class NcclComm(CommunicationBackend):
    backend_id = "nccl"

    def available(self) -> bool:
        try:
            import torch

            return bool(
                torch.cuda.is_available()
                and hasattr(torch.distributed, "is_nccl_available")
                and torch.distributed.is_nccl_available()
            )
        except Exception:  # noqa: BLE001
            return False

    def capabilities(self, devices: tuple[str, ...]) -> CollectiveCapability:
        ok = self.available() and all(d.startswith("cuda_") for d in devices)
        return CollectiveCapability(
            backend_id=self.backend_id,
            available=ok,
            devices=devices if ok else (),
            ops=("allreduce", "broadcast", "reduce", "allgather") if ok else (),
            notes="NVIDIA NCCL" if ok else "NCCL unavailable or devices not CUDA",
        )

    def allreduce(self, tensors: Any, devices: tuple[str, ...]) -> Any:
        if not self.available():
            raise UnsupportedFeatureError("NCCL unavailable")
        result = _pg_allreduce(tensors)
        if result is not None:
            return result
        return HostStagedComm().allreduce(tensors, devices)


class RcclComm(CommunicationBackend):
    backend_id = "rccl"

    def available(self) -> bool:
        try:
            import torch

            return bool(torch.cuda.is_available()) and "rocm" in (torch.version.hip or "").lower()
        except Exception:  # noqa: BLE001
            return False

    def capabilities(self, devices: tuple[str, ...]) -> CollectiveCapability:
        ok = self.available() and all(d.startswith("rocm_") for d in devices)
        return CollectiveCapability(
            backend_id=self.backend_id,
            available=ok,
            devices=devices if ok else (),
            ops=("allreduce", "broadcast") if ok else (),
            notes="AMD RCCL" if ok else "RCCL unavailable",
        )

    def allreduce(self, tensors: Any, devices: tuple[str, ...]) -> Any:
        if not self.available():
            raise UnsupportedFeatureError("RCCL unavailable")
        result = _pg_allreduce(tensors)
        if result is not None:
            return result
        return HostStagedComm().allreduce(tensors, devices)


class OneCclComm(CommunicationBackend):
    backend_id = "oneccl"

    def available(self) -> bool:
        try:
            import oneccl_bindings_for_pytorch  # type: ignore  # noqa: F401

            return True
        except Exception:  # noqa: BLE001
            return False

    def capabilities(self, devices: tuple[str, ...]) -> CollectiveCapability:
        ok = self.available()
        return CollectiveCapability(
            backend_id=self.backend_id,
            available=ok,
            devices=devices if ok else (),
            ops=("allreduce", "broadcast") if ok else (),
            notes="Intel oneCCL" if ok else "oneCCL unavailable",
        )

    def allreduce(self, tensors: Any, devices: tuple[str, ...]) -> Any:
        if not self.available():
            raise UnsupportedFeatureError("oneCCL unavailable")
        result = _pg_allreduce(tensors)
        if result is not None:
            return result
        return HostStagedComm().allreduce(tensors, devices)


class GlooComm(CommunicationBackend):
    backend_id = "gloo"

    def available(self) -> bool:
        try:
            import torch

            return bool(hasattr(torch.distributed, "is_gloo_available") and torch.distributed.is_gloo_available())
        except Exception:  # noqa: BLE001
            return False

    def capabilities(self, devices: tuple[str, ...]) -> CollectiveCapability:
        ok = self.available()
        return CollectiveCapability(
            backend_id=self.backend_id,
            available=ok,
            devices=devices if ok else (),
            ops=("allreduce", "broadcast", "gather") if ok else (),
            notes="Gloo CPU/host collectives" if ok else "Gloo unavailable",
        )

    def allreduce(self, tensors: Any, devices: tuple[str, ...]) -> Any:
        if not self.available():
            raise UnsupportedFeatureError("Gloo unavailable")
        result = _pg_allreduce(tensors)
        if result is not None:
            return result
        return HostStagedComm().allreduce(tensors, devices)


def select_communication_backend(devices: tuple[str, ...]) -> CommunicationBackend:
    """Choose a collective backend from device set; fall back to host staging."""
    candidates: list[CommunicationBackend] = [NcclComm(), RcclComm(), OneCclComm(), GlooComm(), HostStagedComm()]
    for backend in candidates:
        caps = backend.capabilities(devices)
        if caps.available and caps.ops:
            return backend
    return HostStagedComm()
