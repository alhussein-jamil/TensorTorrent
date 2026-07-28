"""Segment compilation: keep whole-machine scheduling separate from kernels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from streamcompiler.backends import backend_by_id
from streamcompiler.backends.base import CompiledRegion, KernelCandidate
from streamcompiler.ir.graph import Instruction, OpCode
from streamcompiler.planner.maximal import Placement


@dataclass
class SegmentExecutable:
    placement: Placement
    region: CompiledRegion


def compile_placement(placement: Placement) -> SegmentExecutable:
    backend = backend_by_id(placement.backend_id)
    if backend is None:
        raise RuntimeError(f"Unknown backend {placement.backend_id}")
    cand = KernelCandidate(
        region_id=placement.region_id,
        device=placement.device,
        backend_id=placement.backend_id,
        kernel_id=placement.kernel_id,
        dtype=placement.dtype,
    )
    region = backend.compile(Instruction(opcode=OpCode.COMPUTE, name=placement.region_id), cand)
    return SegmentExecutable(placement=placement, region=region)


def try_torch_compile_segment(module: Any, example: Any, backend: str = "inductor") -> Any | None:
    """Optionally compile a segment with torch.compile / Inductor when available."""
    try:
        import torch

        compiled = torch.compile(module, backend=backend)
        # Warm smoke call; failures mean fall back to eager segment.
        compiled(example)
        return compiled
    except Exception:
        return None
