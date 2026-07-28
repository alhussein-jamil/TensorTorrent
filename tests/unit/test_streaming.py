"""Weight streaming schedule tests."""

from __future__ import annotations

from pathlib import Path

import torch

from streamcompiler.hardware.discovery import discover_resource_graph
from streamcompiler.ir.graph import HeterogeneousGraph, Instruction, OpCode
from streamcompiler.planner.streaming import synthesize_weight_stream


def test_synthesize_weight_stream_emits_prefetch_evict() -> None:
    machine = discover_resource_graph()
    ir = HeterogeneousGraph(name="stream")
    for i in range(6):
        ir.add_instruction(Instruction(opcode=OpCode.COMPUTE, name=f"l{i}"))
    ir.repeated_blocks = tuple((f"l{i}",) for i in range(6))
    device = next(n for n in machine.compute if n.startswith("cpu_numa_"))
    stages = synthesize_weight_stream(ir, machine, compute_device=device)
    assert stages
    assert any(s.action == "compute" for s in stages)
    assert any(i.opcode == OpCode.PREFETCH for i in ir.instructions)
    assert any(i.opcode == OpCode.EVICT for i in ir.instructions)


def test_pack_header_grows_for_many_tensors(tmp_path: Path) -> None:
    """A fixed header size used to break packs with more than a few dozen tensors."""
    import os

    from streamcompiler.storage.pack import load_pack_manifest, pack_state_dict

    tensors = {f"layer{i}.weight": torch.randn(4, 4) for i in range(300)}
    pack = pack_state_dict(tensors, tmp_path / "many.pack")
    manifest = load_pack_manifest(pack.path)
    assert manifest["tensor_count"] == 300
    assert pack.metadata["header_bytes"] > 4096

    fd = os.open(pack.path, os.O_RDONLY)
    try:
        for name, entry in zip(tensors, manifest["tensors"], strict=True):
            raw = os.pread(fd, entry["nbytes"], entry["offset"])
            restored = torch.frombuffer(bytearray(raw), dtype=torch.float32).reshape(4, 4)
            torch.testing.assert_close(restored, tensors[name])
    finally:
        os.close(fd)
