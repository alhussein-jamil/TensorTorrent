"""Export-free beyond-VRAM path: selection, kwargs/nested I/O, train-mode restore."""

from __future__ import annotations

import torch
import torch.nn as nn

import tensortorrent as tt
from tensortorrent.compile import bakeoff as bakeoff_mod
from tensortorrent.compile import eager_cpu as eager_cpu_mod
from tensortorrent.compile.eager_cpu import (
    estimate_partial_resident_stream_bytes,
    should_prefer_eager_cpu_without_export,
)
from tensortorrent.config import CompileConfig


class _KwargNestedModule(nn.Module):
    """Forces kwargs + nested input/output through the export-free path."""

    def __init__(self, width: int = 1024, depth: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(*[nn.Linear(width, width) for _ in range(depth)])

    def forward(self, payload: dict[str, torch.Tensor], *, scale: float = 1.0) -> dict[str, torch.Tensor]:
        x = payload["x"] * scale
        y = self.net(x)
        return {"y": y, "aux": (y.sum(dim=-1),)}


def _beyond_stream_cfg(vram_bytes: int = 32 << 20) -> CompileConfig:
    """Beyond-VRAM budget that still admits streamed GPU specialization."""
    return CompileConfig(
        use_torch_compile=False,
        measure_regions=False,
        allow_gpu=True,
        allow_cpu=True,
        vram_budget_bytes=int(vram_bytes),
        max_region_nodes=8,
        online_profile_feedback=False,
    )


def test_partial_stream_bytes_leave_headroom() -> None:
    sizes = {f"w{i}": 100 for i in range(10)}
    resident, streamed, selected = estimate_partial_resident_stream_bytes(sizes, budget_bytes=800)
    assert resident + streamed == 1000
    assert streamed > 0
    assert selected
    assert resident < 800


def test_should_prefer_cpu_only_when_confident(monkeypatch) -> None:
    """Large beyond-VRAM state + tiny hoist → non-resident H2D >> CPU → export-free."""
    from benchmarks.suites.workloads import DeepMLP

    model = DeepMLP(1024, 32).eval()
    x = torch.randn(2, 1024)
    monkeypatch.setattr(eager_cpu_mod, "time_eager_call", lambda *a, **k: 0.001)
    prefer, meta = should_prefer_eager_cpu_without_export(model, (x,), {}, _beyond_stream_cfg())
    assert prefer is True, meta
    assert meta["selected"] == "cpu"
    assert meta["param_bytes"] > meta["vram_bytes"]
    assert meta["streamed_param_bytes"] > 0
    assert meta["cpu_fused_s"] * eager_cpu_mod._EXPORT_FREE_CPU_CONFIDENCE < meta["gpu_partial_h2d_predicted_s"]


def test_should_defer_when_partial_gpu_estimate_wins(monkeypatch) -> None:
    """Optimistic non-resident H2D faster than CPU → fall through to bakeoff."""
    from benchmarks.suites.workloads import DeepMLP

    model = DeepMLP(1024, 32).eval()
    x = torch.randn(2, 1024)
    monkeypatch.setattr(eager_cpu_mod, "time_eager_call", lambda *a, **k: 1.0)
    monkeypatch.setattr(
        eager_cpu_mod,
        "estimate_partial_resident_stream_bytes",
        lambda sizes, budget_bytes, transfer_groups=None: (0, 12_000_000, set()),  # ~1ms @ 12GB/s
    )
    prefer, meta = should_prefer_eager_cpu_without_export(model, (x,), {}, _beyond_stream_cfg())
    assert prefer is False
    assert meta["reason"] == "uncertain_or_gpu_favored"
    assert meta["param_bytes"] > meta["vram_bytes"]


def test_should_defer_when_most_params_resident(monkeypatch) -> None:
    """Partial residency leaving tiny stream must not short-circuit to CPU."""
    from benchmarks.suites.workloads import DeepMLP

    model = DeepMLP(1024, 32).eval()
    x = torch.randn(2, 1024)
    pbytes = sum(int(p.numel() * p.element_size()) for p in model.parameters())
    # Beyond VRAM but almost fully resident: tiny stream → GPU H2D looks cheap.
    cfg = _beyond_stream_cfg(vram_bytes=max(64 << 20, pbytes // 2))
    assert pbytes > int(cfg.vram_budget_bytes or 0)
    monkeypatch.setattr(
        eager_cpu_mod,
        "estimate_partial_resident_stream_bytes",
        lambda sizes, budget_bytes, transfer_groups=None: (pbytes - (8 << 20), 8 << 20, {"almost_all"}),
    )
    prefer, meta = should_prefer_eager_cpu_without_export(model, (x,), {}, cfg)
    assert prefer is False, meta
    assert meta["reason"] == "uncertain_or_gpu_favored"


def test_export_free_compile_preserves_train_mode(monkeypatch) -> None:
    from benchmarks.suites.workloads import DeepMLP

    model = DeepMLP(1024, 24)
    model.train()
    assert model.training
    x = torch.randn(2, 1024)
    pbytes = sum(int(p.numel() * p.element_size()) for p in model.parameters())
    monkeypatch.setattr(
        eager_cpu_mod,
        "should_prefer_eager_cpu_without_export",
        lambda *a, **k: (
            True,
            {
                "cpu_fused_s": 0.001,
                "streamed_predicted_s": 1.0,
                "param_bytes": pbytes,
                "selected": "cpu",
                "reason": "deterministic_test",
            },
        ),
    )
    compiled = tt.compile(model, example_inputs=(x,), config=_beyond_stream_cfg())
    try:
        assert model.training is True
        assert compiled.specialized.validation.get("eager_fused_export_free") is True
    finally:
        compiled.close()


def test_export_free_kwargs_and_nested_io() -> None:
    """Beyond-VRAM export-free must preserve kwargs and nested structures."""
    import pytest

    if not torch.cuda.is_available():
        pytest.skip("CUDA required to enter beyond-VRAM export-free gate")

    model = _KwargNestedModule(1024, 32).eval()
    pbytes = sum(int(p.numel() * p.element_size()) for p in model.parameters())
    assert pbytes > (32 << 20)
    x = torch.randn(2, 1024)
    example = (({"x": x},), {"scale": 2.0})
    compiled = tt.compile(model, example_inputs=example, config=_beyond_stream_cfg())
    try:
        assert compiled.specialized.validation.get("eager_fused_export_free") is True
        with torch.inference_mode():
            out = compiled({"x": x}, scale=2.0)
            ref = model({"x": x}, scale=2.0)
        assert set(out) == set(ref)
        assert torch.allclose(out["y"], ref["y"])
        assert torch.allclose(out["aux"][0], ref["aux"][0])
    finally:
        compiled.close()


def test_beyond_vram_cpu_win_selects_auto_cpu() -> None:
    """Beyond VRAM + transfer-dominated H2D estimate → export-free auto CPU."""
    import pytest
    from benchmarks.suites.workloads import DeepMLP

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    model = DeepMLP(1024, 32).eval()
    x = torch.randn(2, 1024)
    compiled = tt.compile(model, example_inputs=(x,), config=_beyond_stream_cfg())
    try:
        assert compiled.specialized.validation.get("eager_fused_export_free") is True
        devices = list(compiled.specialized.plan.devices_used)
        assert devices and all(str(d).startswith("cpu") for d in devices)
        assert compiled.specialized.validation.get("baseline_guard_selected") == "cpu"
    finally:
        compiled.close()


def test_should_prefer_gpu_when_state_fits_hoist_budget() -> None:
    from tensortorrent.compile.eager_cpu import should_prefer_eager_gpu_without_export

    model = nn.Linear(8, 4).eval()
    x = torch.randn(2, 8)
    cfg = CompileConfig(use_torch_compile=False, measure_regions=False, allow_gpu=True, allow_cpu=True)
    prefer, meta = should_prefer_eager_gpu_without_export(model, (x,), {}, cfg)
    if not torch.cuda.is_available():
        assert prefer is False
        assert meta["reason"] == "cuda_unavailable"
        return
    assert prefer is True, meta
    assert meta["reason"] == "fits_vram"
    assert meta["selected"] == "gpu"


def test_should_reject_gpu_export_free_when_over_hoist_budget() -> None:
    from tensortorrent.compile.eager_cpu import should_prefer_eager_gpu_without_export

    model = nn.Linear(8, 4).eval()
    x = torch.randn(2, 8)
    cfg = CompileConfig(
        use_torch_compile=False,
        allow_gpu=True,
        vram_budget_bytes=64,
    )
    prefer, meta = should_prefer_eager_gpu_without_export(model, (x,), {}, cfg)
    assert prefer is False
    assert meta["reason"] in {"exceeds_hoist_budget", "cuda_unavailable"}


def test_fit_vram_compile_skips_export_on_cuda() -> None:
    import pytest

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    model = nn.Linear(32, 16).eval()
    x = torch.randn(4, 32)
    with torch.inference_mode():
        ref = model(x).clone()
    compiled = tt.compile(model, example_inputs=(x,), config=CompileConfig(use_torch_compile=False))
    try:
        assert compiled.specialized.validation.get("eager_fused_gpu") is True
        assert compiled.specialized.validation.get("eager_fused_export_free") is True
        with torch.inference_mode():
            out = compiled(x)
        assert out.device.type == "cuda"
        assert torch.allclose(out.cpu(), ref, atol=1e-5)
        assert next(model.parameters()).device.type == "cpu"
    finally:
        compiled.close()


def test_zero_param_module_skips_gpu_export_free() -> None:
    from tensortorrent.compile.eager_cpu import should_prefer_eager_gpu_without_export

    class Identity(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    prefer, meta = should_prefer_eager_gpu_without_export(
        Identity().eval(),
        (torch.randn(2, 3),),
        {},
        CompileConfig(use_torch_compile=False, allow_gpu=True),
    )
    assert prefer is False
    assert meta["reason"] in {"no_parameters", "cuda_unavailable"}


def test_export_free_gpu_save_roundtrip(tmp_path) -> None:
    import pytest

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    model = nn.Linear(8, 4).eval()
    x = torch.randn(2, 8)
    with torch.inference_mode():
        ref = model(x).clone()
    compiled = tt.compile(model, example_inputs=(x,), config=CompileConfig(use_torch_compile=False))
    try:
        assert compiled.specialized.validation.get("eager_fused_gpu") is True
        compiled.save(tmp_path / "saved")
        assert (tmp_path / "saved" / "exported.pt2").exists()
        reloaded = tt.load_compiled(tmp_path / "saved")
        try:
            torch.testing.assert_close(reloaded(x).cpu(), ref)
        finally:
            reloaded.close()
    finally:
        compiled.close()
    model = nn.Linear(8, 4).eval()
    x = torch.randn(2, 8)
    compiled = tt.compile(
        model,
        example_inputs=(x,),
        artifact_dir=tmp_path / "bundle",
        config=CompileConfig(use_torch_compile=False, allow_gpu=False),
    )
    try:
        assert not compiled.specialized.validation.get("eager_fused_gpu")
        assert compiled.specialized.schedule is not None
    finally:
        compiled.close()


def test_beyond_vram_gpu_win_falls_through_then_auto_cuda(monkeypatch) -> None:
    """When partial-H2D estimate beats CPU, auto must not take export-free CPU.

    After fall-through, force bakeoff to keep the streamed accelerator plan so
    auto lands on GPU (proves the gate no longer short-circuits to CPU).
    """
    import pytest
    from benchmarks.suites.workloads import DeepMLP

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    monkeypatch.setattr(
        eager_cpu_mod,
        "should_prefer_eager_cpu_without_export",
        lambda *a, **k: (False, {"reason": "uncertain_or_gpu_favored", "checked": True}),
    )
    monkeypatch.setattr(
        eager_cpu_mod,
        "should_prefer_eager_gpu_without_export",
        lambda *a, **k: (False, {"reason": "exceeds_hoist_budget", "checked": True}),
    )
    # Defeat early "skip streamed specialize" (cpu * 1.5 < predicted) and
    # measured prefer_cpu_baseline so bakeoff keeps the accelerator plan.
    monkeypatch.setattr(bakeoff_mod, "time_eager_module", lambda *a, **k: 10.0)
    monkeypatch.setattr(bakeoff_mod, "prefer_cpu_baseline", lambda **kw: False)
    monkeypatch.setattr(bakeoff_mod, "safe_time_executor", lambda *a, **k: bakeoff_mod.BakeoffTiming(0.001, True))

    model = DeepMLP(256, 12).eval()
    x = torch.randn(2, 256)
    pbytes = sum(int(p.numel() * p.element_size()) for p in model.parameters())
    compiled = tt.compile(
        model,
        example_inputs=(x,),
        config=_beyond_stream_cfg(vram_bytes=max(pbytes // 2, 1 << 20)),
    )
    try:
        assert not compiled.specialized.validation.get("eager_fused_export_free")
        devices = {str(b.device) for b in compiled.specialized.bindings.values()}
        assert any(d.startswith("cuda_") for d in devices), devices
        selected = compiled.specialized.validation.get("baseline_guard_selected")
        assert selected in {"streamed", "accelerator", "gpu_prefix_overflow"} or any(
            d.startswith("cuda_") for d in devices
        )
    finally:
        compiled.close()
