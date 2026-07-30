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


class HostStagedComm(CommunicationBackend):
    """Always-available fallback when vendor collectives cannot span devices."""

    backend_id = "host_staged"

    def available(self) -> bool:
        return True

    def capabilities(self, devices: tuple[str, ...]) -> CollectiveCapability:
        return CollectiveCapability(
            backend_id=self.backend_id,
            available=True,
            devices=devices,
            ops=("broadcast", "reduce", "allreduce", "gather", "scatter"),
            notes="Copies via host memory; slower but mixed-vendor safe",
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
        raise UnsupportedFeatureError(
            "NCCL allreduce is not wired through StreamCompiler yet; use host_staged or implement "
            f"a real torch.distributed collective for devices={devices}"
        )


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
            notes="RCCL" if ok else "RCCL unavailable",
        )

    def allreduce(self, tensors: Any, devices: tuple[str, ...]) -> Any:
        if not self.available():
            raise UnsupportedFeatureError("RCCL unavailable")
        raise UnsupportedFeatureError(f"RCCL allreduce is not wired through StreamCompiler yet; devices={devices}")


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
            notes="oneCCL" if ok else "oneCCL unavailable",
        )

    def allreduce(self, tensors: Any, devices: tuple[str, ...]) -> Any:
        if not self.available():
            raise UnsupportedFeatureError("oneCCL unavailable")
        raise UnsupportedFeatureError(f"oneCCL allreduce is not wired through StreamCompiler yet; devices={devices}")


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
        import torch

        # When a process group is already initialized, use it (multi-node ready).
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            if isinstance(tensors, torch.Tensor):
                handle = tensors.detach().clone()
                torch.distributed.all_reduce(handle)
                return handle
            raise UnsupportedFeatureError("Gloo process-group allreduce expects a single tensor")
        # Single-process / multi-peer host fallback used for multi-node bring-up tests.
        return HostStagedComm().allreduce(tensors, devices)


def select_communication_backend(devices: tuple[str, ...]) -> CommunicationBackend:
    """Choose a collective backend from device set; fall back to host staging."""
    candidates: list[CommunicationBackend] = [NcclComm(), RcclComm(), OneCclComm(), GlooComm(), HostStagedComm()]
    for backend in candidates:
        caps = backend.capabilities(devices)
        if caps.available and caps.ops:
            return backend
    return HostStagedComm()
