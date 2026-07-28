"""Runtime service tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from streamcompiler.errors import RuntimePlanError
from streamcompiler.hardware.discovery import discover_resource_graph
from streamcompiler.runtime import EventPool, IoExecutor, TensorDirectory, TieredAllocator
from streamcompiler.runtime.executor import ResidentCopy, TensorRecord
from streamcompiler.storage.pack import load_pack_manifest, pack_state_dict


def test_tiered_allocator_respects_capacity() -> None:
    machine = discover_resource_graph()
    alloc = TieredAllocator(machine)
    mem_name = next(iter(machine.memory))
    alloc.allocate(mem_name, 1024)
    assert alloc.used()[mem_name] == 1024
    alloc.release(mem_name, 1024)
    assert alloc.used()[mem_name] == 0


def test_tensor_directory_tracks_copies() -> None:
    directory = TensorDirectory()
    directory.register(
        TensorRecord(
            tensor_id="w",
            home="numa_ram_0",
            copies=[ResidentCopy(memory="numa_ram_0", version=0, nbytes=16)],
        )
    )
    assert directory.get("w").home == "numa_ram_0"
    directory.invalidate_stale("w", 1)
    assert directory.get("w").copies == []
    with pytest.raises(RuntimePlanError, match="Unknown tensor"):
        directory.get("missing")


def test_event_pool_reports_unknown_events() -> None:
    pool = EventPool()
    pool.record("done:region_0")
    assert pool.wait("done:region_0", timeout=0.1) is True
    with pytest.raises(RuntimePlanError, match="Unknown event"):
        pool.wait("done:region_9")


def test_io_executor_reads_real_blocks(tmp_path: Path) -> None:
    tensor = torch.arange(16, dtype=torch.float32)
    pack = pack_state_dict({"w": tensor}, tmp_path / "m.pack")
    block = load_pack_manifest(pack.path)["tensors"][0]

    io = IoExecutor()
    result = io.prefetch(str(pack.path), block["offset"], block["nbytes"])
    assert result["status"] == "read"
    assert result["nbytes"] == block["nbytes"]

    raw = io.read_block(str(pack.path), block["offset"], block["nbytes"])
    restored = torch.frombuffer(bytearray(raw), dtype=torch.float32)
    torch.testing.assert_close(restored, tensor)


def test_io_executor_rejects_short_reads(tmp_path: Path) -> None:
    path = tmp_path / "small.bin"
    path.write_bytes(b"12345")
    with pytest.raises(RuntimePlanError, match="Short read"):
        IoExecutor().read_block(str(path), 0, 64)


def test_region_worker_threads_run_in_inference_mode() -> None:
    """torch.inference_mode is thread-local, so pool workers must opt in themselves."""
    from streamcompiler.parallel import inference_thread_pool

    with inference_thread_pool(max_workers=2, thread_name_prefix="test-region") as pool:
        modes = list(pool.map(lambda _: torch.is_inference_mode_enabled(), range(4)))
    assert modes == [True] * 4


def test_default_thread_pool_would_not_inherit_inference_mode() -> None:
    """Guards the reason the initializer exists: plain pools drop the mode."""
    from concurrent.futures import ThreadPoolExecutor

    with torch.inference_mode(), ThreadPoolExecutor(max_workers=1) as pool:
        assert torch.is_inference_mode_enabled() is True
        assert pool.submit(torch.is_inference_mode_enabled).result() is False
