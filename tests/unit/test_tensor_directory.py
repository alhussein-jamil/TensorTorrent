"""TensorDirectory residency state machine: the runtime's source of truth for
which logical tensors have valid copies where, and whether a transfer can be
reused instead of duplicated."""

from __future__ import annotations

import threading
import time

import pytest
import torch

from streamcompiler.errors import RuntimePlanError
from streamcompiler.ir.graph import OpCode
from streamcompiler.runtime.schedule import MemoryTier, PlanInstruction
from streamcompiler.runtime.tensor_directory import TensorDirectory, TensorState
from streamcompiler.runtime.transfers import execute_transfer_instruction


def test_already_resident_tensor_is_reused_without_transfer() -> None:
    directory = TensorDirectory()
    directory.mark_produced("t0", location="cpu_numa_0", tier=MemoryTier.SYSTEM_RAM, value=torch.randn(4))
    assert directory.has_copy_at("t0", "cpu_numa_0")
    inst = PlanInstruction(
        opcode=OpCode.TRANSFER,
        name="transfer::t0",
        resource="copy_engine",
        inputs=("t0",),
        source="cpu_numa_0",
        destination="cpu_numa_0",
        transfer_backend="host_memcpy",
    )
    _out, result = execute_transfer_instruction(inst, torch.randn(4), directory)
    assert result.backend == "elided_duplicate"
    assert result.nbytes == 0


def test_disk_to_ram_transition_materializes_a_real_copy() -> None:
    directory = TensorDirectory()
    record = directory.materialize("w0", location="disk_pack", tier=MemoryTier.DISK, nbytes=64, value=torch.randn(16))
    assert record.state is TensorState.ON_DISK
    loaded = torch.randn(16)
    inst = PlanInstruction(
        opcode=OpCode.LOAD,
        name="load::w0",
        resource="cpu_numa_0",
        inputs=("w0",),
        source="disk_pack",
        destination="cpu_numa_0",
        transfer_backend="disk_pread",
    )
    out, result = execute_transfer_instruction(inst, loaded, directory, disk_loader=lambda v: v)
    assert result.backend == "disk_pread"
    assert torch.equal(out, loaded)
    assert directory.has_copy_at("w0", "cpu_numa_0")
    assert directory.get("w0").state is TensorState.IN_RAM


def test_in_progress_transfer_is_joined_not_duplicated() -> None:
    """Two concurrent consumers requesting the same tensor at the same destination
    must collapse into one transfer; the second joins instead of copying again."""
    directory = TensorDirectory()
    directory.mark_produced("t0", location="src", tier=MemoryTier.SYSTEM_RAM, value=torch.randn(4))
    started = threading.Event()
    release = threading.Event()
    calls = []

    class SlowLoader:
        def __call__(self, value: torch.Tensor) -> torch.Tensor:
            calls.append(1)
            started.set()
            release.wait(timeout=5)
            return value.clone()

    inst = PlanInstruction(
        opcode=OpCode.TRANSFER,
        name="transfer::t0",
        resource="copy_engine",
        inputs=("t0",),
        source="src",
        destination="dst",
        transfer_backend="disk_pread",
    )
    loader = SlowLoader()
    results: list[tuple] = []

    def run_transfer() -> None:
        results.append(execute_transfer_instruction(inst, torch.randn(4), directory, disk_loader=loader))

    t1 = threading.Thread(target=run_transfer)
    t1.start()
    assert started.wait(timeout=5), "first transfer never started"
    # First transfer is in flight (not yet completed); a second request to the
    # same destination must join it rather than starting a second real copy.
    t2 = threading.Thread(target=run_transfer)
    t2.start()
    time.sleep(0.05)
    release.set()
    t1.join(timeout=5)
    t2.join(timeout=5)

    assert len(calls) == 1, "duplicate real transfer performed while one was already in flight"
    backends = {r[1].backend for r in results}
    assert "disk_pread" in backends
    assert "joined_in_progress_transfer" in backends


def test_stale_copy_invalidated_after_mutation() -> None:
    directory = TensorDirectory()
    directory.ensure("p0", mutable=True)
    directory.mark_produced("p0", location="cpu_numa_0", tier=MemoryTier.SYSTEM_RAM, value=torch.randn(4))
    directory.materialize("p0", location="cpu_numa_0_copy", tier=MemoryTier.SYSTEM_RAM, nbytes=16)
    assert len(directory.locations("p0")) == 2
    version_before = directory.get("p0").version
    directory.mutate("p0")
    record = directory.get("p0")
    assert record.version == version_before + 1
    assert len(record.valid_copies) == 1, "mutation must invalidate all but one canonical copy"


def test_mutation_of_immutable_tensor_is_rejected() -> None:
    directory = TensorDirectory()
    directory.ensure("a0", mutable=False)
    directory.mark_produced("a0", location="cpu_numa_0", tier=MemoryTier.SYSTEM_RAM, value=torch.randn(4))
    with pytest.raises(RuntimePlanError):
        directory.mutate("a0")


def test_release_after_final_consumer_frees_copies() -> None:
    directory = TensorDirectory()
    directory.mark_produced("t0", location="cpu_numa_0", tier=MemoryTier.SYSTEM_RAM, value=torch.randn(4))
    directory.add_consumer("t0")
    directory.add_consumer("t0")
    assert directory.release("t0") is False, "must not release while consumers remain"
    assert directory.finish_consumer("t0") == 1
    assert directory.release("t0") is False
    assert directory.finish_consumer("t0") == 0
    assert directory.release("t0") is True
    assert directory.get("t0").state is TensorState.RELEASED
    assert directory.locations("t0") == ()


def test_retention_across_several_consumers() -> None:
    directory = TensorDirectory()
    directory.mark_produced("t0", location="cpu_numa_0", tier=MemoryTier.SYSTEM_RAM, value=torch.randn(4))
    for _ in range(3):
        directory.add_consumer("t0")
    for _ in range(2):
        directory.finish_consumer("t0")
        assert directory.has_copy_at("t0", "cpu_numa_0"), "copy must persist while any consumer remains"
    directory.finish_consumer("t0")
    directory.release("t0")
    assert not directory.has_copy_at("t0", "cpu_numa_0")
