"""Disk streaming under a RAM budget, with real reads and double buffering."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

import streamcompiler as sc
from streamcompiler.errors import MemoryCapacityError
from streamcompiler.runtime.tensor_store import (
    ResidentParameterStore,
    StreamingParameterStore,
)
from streamcompiler.storage.pack import pack_state_dict


class Deep(nn.Module):
    """Enough distinct weights that a small budget forces eviction and re-reads."""

    def __init__(self, width: int = 64, layers: int = 6) -> None:
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(width, width) for _ in range(layers))
        self.head = nn.Linear(width, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = torch.relu(layer(x))
        return self.head(x)


def _streaming_config(budget: int, prefetch: int = 1) -> sc.CompileConfig:
    return sc.CompileConfig(ram_budget_bytes=budget, prefetch_distance=prefetch)


def test_resident_store_is_used_when_weights_fit() -> None:
    compiled = sc.compile(Deep().eval(), (torch.randn(2, 64),))
    stats = compiled._executor.parameter_store.stats()
    assert stats["kind"] == "resident"
    assert stats["resident_bytes"] > 0


def test_ram_budget_triggers_disk_streaming_and_matches_eager() -> None:
    model = Deep().eval()
    x = torch.randn(3, 64)
    with torch.no_grad():
        expected = model(x)
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    compiled = sc.compile(model, (x,), config=_streaming_config(total // 4))
    stats = compiled._executor.parameter_store.stats()
    assert stats["kind"] == "streaming"
    assert stats["budget_bytes"] < total

    torch.testing.assert_close(compiled(x), expected)
    stats = compiled._executor.parameter_store.stats()
    assert stats["reads"] > 0, "streaming store performed no reads"
    assert stats["bytes_read"] > 0
    assert stats["peak_resident_bytes"] <= stats["budget_bytes"]


def test_streaming_store_reads_real_bytes_from_the_pack(tmp_path: Path) -> None:
    tensors = {"a.weight": torch.randn(16, 16), "b.weight": torch.randn(8, 8)}
    pack = pack_state_dict(tensors, tmp_path / "m.pack")
    bindings = {"env_a": "a.weight", "env_b": "b.weight"}
    store = StreamingParameterStore(pack.path, bindings, budget_bytes=1 << 20)
    try:
        torch.testing.assert_close(store.acquire("env_a"), tensors["a.weight"])
        torch.testing.assert_close(store.acquire("env_b"), tensors["b.weight"])
        stats = store.stats()
        assert stats["reads"] == 2
        assert stats["bytes_read"] == sum(t.numel() * t.element_size() for t in tensors.values())
    finally:
        store.close()


def test_streaming_store_evicts_under_budget(tmp_path: Path) -> None:
    tensors = {f"w{i}": torch.randn(32, 32) for i in range(4)}
    block = 32 * 32 * 4
    pack = pack_state_dict(tensors, tmp_path / "m.pack")
    bindings = {f"env{i}": f"w{i}" for i in range(4)}
    store = StreamingParameterStore(pack.path, bindings, budget_bytes=block * 2)
    try:
        for i in range(4):
            name = f"env{i}"
            torch.testing.assert_close(store.acquire(name), tensors[f"w{i}"])
            store.release((name,))
        stats = store.stats()
        assert stats["evictions"] > 0
        assert stats["peak_resident_bytes"] <= block * 2
    finally:
        store.close()


def test_streaming_store_rejects_budget_smaller_than_one_block(tmp_path: Path) -> None:
    pack = pack_state_dict({"w": torch.randn(64, 64)}, tmp_path / "m.pack")
    with pytest.raises(MemoryCapacityError, match="cannot hold the largest"):
        StreamingParameterStore(pack.path, {"env": "w"}, budget_bytes=16)


def test_prefetch_performs_real_io_ahead_of_use(tmp_path: Path) -> None:
    tensors = {f"w{i}": torch.randn(64, 64) for i in range(4)}
    pack = pack_state_dict(tensors, tmp_path / "m.pack")
    bindings = {f"env{i}": f"w{i}" for i in range(4)}
    store = StreamingParameterStore(pack.path, bindings, budget_bytes=1 << 20)
    try:
        store.prefetch(tuple(bindings))
        for i in range(4):
            torch.testing.assert_close(store.acquire(f"env{i}"), tensors[f"w{i}"])
        stats = store.stats()
        assert stats["prefetch_submitted"] == 4
        assert stats["reads"] == 4
        assert stats["bytes_read"] == sum(t.numel() * t.element_size() for t in tensors.values())
        # Every acquire was satisfied by cache or by a prefetch already in flight.
        assert stats["cache_misses"] == 0
    finally:
        store.close()


def test_double_buffering_overlaps_the_next_region_load() -> None:
    """While a region computes, the next region's weights are already being read."""
    model = Deep(width=128, layers=6).eval()
    x = torch.randn(8, 128)
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    compiled = sc.compile(model, (x,), config=_streaming_config(total // 3, prefetch=1))
    with torch.no_grad():
        expected = model(x)
    torch.testing.assert_close(compiled(x), expected)
    stats = compiled._executor.parameter_store.stats()
    assert stats["kind"] == "streaming"
    assert stats["prefetch_submitted"] > 0, "no prefetch was issued"
    overlapped = stats["prefetch_hits"] + stats["cache_hits"] + stats["waits_for_prefetch"]
    assert overlapped > 0, "prefetched blocks were never consumed"


def test_prefetch_can_be_disabled() -> None:
    model = Deep().eval()
    x = torch.randn(2, 64)
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    compiled = sc.compile(model, (x,), config=_streaming_config(total // 2, prefetch=0))
    with torch.no_grad():
        expected = model(x)
    torch.testing.assert_close(compiled(x), expected)
    stats = compiled._executor.parameter_store.stats()
    assert stats["prefetch_submitted"] == 0
    assert stats["reads"] > 0


def test_streaming_survives_repeated_calls() -> None:
    model = Deep().eval()
    x = torch.randn(2, 64)
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    compiled = sc.compile(model, (x,), config=_streaming_config(total // 4))
    with torch.no_grad():
        expected = model(x)
    for _ in range(3):
        torch.testing.assert_close(compiled(x), expected)
    stats = compiled._executor.parameter_store.stats()
    assert stats["peak_resident_bytes"] <= stats["budget_bytes"]


def test_resident_store_reports_unknown_parameters() -> None:
    from streamcompiler.errors import StorageError

    store = ResidentParameterStore({"a": torch.zeros(2)})
    with pytest.raises(StorageError, match="Unknown parameter"):
        store.acquire("missing")
