"""Distinct virtual-device tensor handle for schedule-managed mock accelerators.

Host ``torch.Tensor`` values must not be silently treated as device-resident.
Transfers to a virtual accelerator wrap the payload; compute unwraps; host
consumers require an explicit Transfer back.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from streamcompiler.errors import RuntimePlanError


@dataclass
class VirtualDeviceTensor:
    """Host-backed payload labelled as resident on a virtual device resource."""

    payload: torch.Tensor
    device_id: str
    nbytes: int
    allocation_key: str
    simulated: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.payload, torch.Tensor):
            raise TypeError("VirtualDeviceTensor.payload must be a torch.Tensor")
        if self.nbytes <= 0:
            self.nbytes = int(self.payload.numel() * self.payload.element_size())
        if not self.allocation_key:
            storage = self.payload.untyped_storage()
            self.allocation_key = f"vdev::{self.device_id}::{int(storage.data_ptr())}::{int(storage.nbytes())}"

    def to_host(self) -> torch.Tensor:
        return self.payload

    def clone(self) -> VirtualDeviceTensor:
        return VirtualDeviceTensor(
            payload=self.payload.detach().clone(),
            device_id=self.device_id,
            nbytes=self.nbytes,
            allocation_key="",
            simulated=True,
        )


def wrap_virtual(value: Any, device_id: str) -> VirtualDeviceTensor:
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


def unwrap_for_compute(value: Any, *, resource: str) -> Any:
    """Accept host tensors on CPU resources; require VirtualDeviceTensor on mock devices."""
    mock = "mock" in resource.lower()
    if mock:
        if isinstance(value, VirtualDeviceTensor):
            return value.payload
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
