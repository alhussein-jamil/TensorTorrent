"""Native virtual-device tensor handles for schedule-managed mock accelerators.

Host ``torch.Tensor`` values must not be silently treated as device-resident.
Transfers to a virtual accelerator allocate a native virtual buffer and store
an opaque handle; compute reads those bytes back for the region body.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from streamcompiler.errors import RuntimePlanError


@dataclass
class VirtualDeviceTensor:
    """Opaque native virtual-buffer handle labelled as device-resident.

    ``payload`` is a host staging view used only for region math; the authoritative
    device copy is ``native_buffer_id`` on the execution context's VirtualBackend.
    """

    payload: torch.Tensor
    device_id: str
    nbytes: int
    allocation_key: str
    simulated: bool = True
    native_buffer_id: int | None = None
    _native_ctx: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.payload, torch.Tensor):
            raise TypeError("VirtualDeviceTensor.payload must be a torch.Tensor")
        if self.nbytes <= 0:
            self.nbytes = int(self.payload.numel() * self.payload.element_size())
        if not self.allocation_key:
            if self.native_buffer_id is not None:
                self.allocation_key = f"vdev::{self.device_id}::buf{int(self.native_buffer_id)}"
            else:
                storage = self.payload.untyped_storage()
                self.allocation_key = f"vdev::{self.device_id}::{int(storage.data_ptr())}::{int(storage.nbytes())}"

    def to_host(self) -> torch.Tensor:
        if self.native_buffer_id is not None and self._native_ctx is not None:
            raw = self._native_ctx.virtual_buffer_to_bytes(self.device_id, int(self.native_buffer_id))
            # PyBytes is read-only; one bytearray for frombuffer. Retain buf — no clone.
            buf = raw if isinstance(raw, bytearray) else bytearray(raw)
            tensor = torch.frombuffer(buf, dtype=self.payload.dtype).reshape(self.payload.shape)
            tensor._sc_host_buf = buf  # type: ignore[attr-defined]
            return tensor
        return self.payload

    def clone(self) -> VirtualDeviceTensor:
        return VirtualDeviceTensor(
            payload=self.payload.detach().clone(),
            device_id=self.device_id,
            nbytes=self.nbytes,
            allocation_key="",
            simulated=True,
            native_buffer_id=self.native_buffer_id,
            _native_ctx=self._native_ctx,
        )


def wrap_virtual(value: Any, device_id: str) -> VirtualDeviceTensor:
    """Legacy host-staging wrap for bench-only SimulatedDeviceTransfer.

    Prefer :func:`wrap_virtual_native` on the production path.
    """
    if isinstance(value, VirtualDeviceTensor):
        if value.device_id != device_id:
            raise RuntimePlanError(
                f"VirtualDeviceTensor on {value.device_id!r} cannot move to {device_id!r} without Transfer"
            )
        return value
    if not isinstance(value, torch.Tensor):
        raise RuntimePlanError(f"Cannot place non-tensor {type(value).__name__} on virtual device {device_id}")
    return VirtualDeviceTensor(
        payload=value.detach(),
        device_id=device_id,
        nbytes=int(value.numel() * value.element_size()),
        allocation_key="",
        simulated=True,
    )


def wrap_virtual_native(value: Any, device_id: str, native_ctx: Any) -> VirtualDeviceTensor:
    """Allocate a native virtual buffer and store host bytes there (not a host alias)."""
    if isinstance(value, VirtualDeviceTensor):
        if value.device_id != device_id:
            raise RuntimePlanError(
                f"VirtualDeviceTensor on {value.device_id!r} cannot move to {device_id!r} without Transfer"
            )
        return value
    if not isinstance(value, torch.Tensor):
        raise RuntimePlanError(f"Cannot place non-tensor {type(value).__name__} on virtual device {device_id}")
    host = value.detach().contiguous().cpu()
    raw = bytes(host.numpy().tobytes())
    buf_id = int(native_ctx.virtual_buffer_from_bytes(device_id, raw))
    return VirtualDeviceTensor(
        payload=host,
        device_id=device_id,
        nbytes=len(raw),
        allocation_key=f"vdev::{device_id}::buf{buf_id}",
        simulated=True,
        native_buffer_id=buf_id,
        _native_ctx=native_ctx,
    )


def unwrap_for_compute(value: Any, *, resource: str) -> Any:
    """Accept host tensors on CPU resources; require VirtualDeviceTensor on mock devices."""
    mock = "mock" in resource.lower()
    if mock:
        if isinstance(value, VirtualDeviceTensor):
            return value.to_host()
        raise RuntimePlanError(
            f"Compute on {resource}: expected VirtualDeviceTensor, got {type(value).__name__}; "
            "schedule must Transfer host→device before Compute (simulated)"
        )
    if isinstance(value, VirtualDeviceTensor):
        raise RuntimePlanError(
            f"Compute on {resource}: VirtualDeviceTensor still device-resident; "
            "schedule must Transfer device→host first (simulated)"
        )
    return value
