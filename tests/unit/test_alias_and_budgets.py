"""Alias analysis and activation / VRAM budget knobs."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

import streamcompiler as sc
from streamcompiler.analysis.alias import run_alias_analysis
from streamcompiler.config import CompileConfig
from streamcompiler.ir.graph import HeterogeneousGraph, TensorMeta
from streamcompiler.runtime.tensor_store import StreamingParameterStore
from streamcompiler.storage.pack import pack_state_dict


def test_alias_analysis_groups_shared_storage() -> None:
    graph = HeterogeneousGraph(
        name="alias",
        metadata={"state_bindings": {"get_attr_0": "shared.weight", "get_attr_1": "shared.weight"}},
    )
    graph.add_tensor(TensorMeta("get_attr_0", (4, 4), "float32", size_bytes=64, kind="parameter"))
    graph.add_tensor(TensorMeta("get_attr_1", (4, 4), "float32", size_bytes=64, kind="parameter"))
    graph.add_tensor(TensorMeta("act", (4,), "float32", size_bytes=16, kind="activation"))
    alias = run_alias_analysis(graph)
    assert alias.groups["get_attr_0"] == alias.groups["get_attr_1"] == "storage::shared.weight"
    assert alias.groups["act"] == "act"
    assert graph.tensors["get_attr_0"].storage_id == "shared.weight"


def test_streaming_cache_dedupes_aliased_env_names(tmp_path: Path) -> None:
    weight = torch.randn(8, 8)
    path = tmp_path / "model.pack"
    pack_state_dict({"shared.weight": weight}, path)
    store = StreamingParameterStore(
        path,
        {"left": "shared.weight", "right": "shared.weight"},
        budget_bytes=weight.numel() * weight.element_size() * 2,
    )
    try:
        a = store.acquire("left")
        b = store.acquire("right")
        assert a.data_ptr() == b.data_ptr()
        stats = store.stats()
        assert stats["block_count"] == 1
        assert stats["duplicate_reads_avoided"] >= 1
        assert stats["peak_resident_bytes"] == weight.numel() * weight.element_size()
        store.release(("left", "right"))
    finally:
        store.close()


def test_plan_explain_marks_measured_placements() -> None:
    model = nn.Linear(16, 8).eval()
    x = torch.randn(2, 16)
    compiled = sc.compile(model, (x,), config=CompileConfig(measure_regions=True))
    text = compiled.explain()
    assert "(measured)" in text or "(prior)" in text
    assert isinstance(compiled.specialized.profile.get("transfers"), dict)
    compiled.close()


def test_activation_budget_enables_runtime_spill_note() -> None:
    model = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 16)).eval()
    x = torch.randn(8, 64)
    compiled = sc.compile(model, (x,), config=CompileConfig(activation_budget_bytes=1, use_torch_compile=False))
    try:
        notes = " ".join(compiled.specialized.plan.notes)
        assert "schedule activation spill" in notes or "activation_peak" in notes
        assert compiled.config.activation_budget_bytes == 1
        assert compiled.specialized.schedule is not None
        assert any("activation" in n for n in compiled.specialized.plan.notes)
    finally:
        compiled.close()


def test_vram_budget_serializes_in_config() -> None:
    cfg = CompileConfig(vram_budget_bytes=1 << 20, activation_budget_bytes=1 << 18)
    roundtrip = CompileConfig.from_json_dict(cfg.to_json_dict())
    assert roundtrip.vram_budget_bytes == 1 << 20
    assert roundtrip.activation_budget_bytes == 1 << 18
