"""Auto mode must prefer measured fused CPU when the accelerator plan is slower."""

from __future__ import annotations

import time

import pytest
import torch

import tensortorrent as tt
from tensortorrent.compile.bakeoff import bindings_use_accelerator as _bindings_use_accelerator
from tensortorrent.compile.bakeoff import prefer_cpu_baseline as _prefer_cpu_baseline
from tensortorrent.compile.bakeoff import select_beyond_vram_winner as _select_beyond_vram_winner


def test_bindings_use_accelerator_detects_cuda_device() -> None:
    class _B:
        def __init__(self, backend_id: str, device: str) -> None:
            self.backend_id = backend_id
            self.device = device

    assert _bindings_use_accelerator({"r0": _B("cuda", "cuda_gpu_0")})  # type: ignore[arg-type]
    assert not _bindings_use_accelerator({"r0": _B("cpu", "cpu_numa_0")})  # type: ignore[arg-type]


def test_prefer_cpu_baseline_hysteresis() -> None:
    assert _prefer_cpu_baseline(cpu_s=1.0, streamed_s=1.0)
    assert _prefer_cpu_baseline(cpu_s=1.01, streamed_s=1.0)
    assert not _prefer_cpu_baseline(cpu_s=1.03, streamed_s=1.0)


def test_select_beyond_vram_winner_three_way() -> None:
    assert _select_beyond_vram_winner(cpu_s=1.0, streamed_s=2.0, overflow_s=3.0) == "cpu"
    assert _select_beyond_vram_winner(cpu_s=2.0, streamed_s=1.0, overflow_s=3.0) == "streamed"
    assert _select_beyond_vram_winner(cpu_s=2.0, streamed_s=3.0, overflow_s=1.0) == "gpu_prefix_overflow"
    # Equal accel times prefer static overflow over streamed H2D.
    assert _select_beyond_vram_winner(cpu_s=2.0, streamed_s=1.0, overflow_s=1.0) == "gpu_prefix_overflow"
    assert _select_beyond_vram_winner(cpu_s=1.01, streamed_s=1.0, overflow_s=float("inf")) == "cpu"
    assert _select_beyond_vram_winner(cpu_s=float("inf"), streamed_s=1.0, overflow_s=2.0) == "streamed"


def test_beyond_vram_bakeoff_rejects_multiregion_cpu_stream() -> None:
    """Streamed bakeoff candidate must be accelerator-bound, not multi-region CPU."""
    import pytest
    from benchmarks.suites.workloads import DeepMLP

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    model = DeepMLP(512, 16).eval()
    x = torch.randn(2, 512)
    compiled = tt.compile(
        model,
        example_inputs=(x,),
        config=tt.CompileConfig(
            use_torch_compile=False,
            measure_regions=False,
            allow_gpu=True,
            allow_cpu=True,
            vram_budget_bytes=8 << 20,
            max_region_nodes=4,
        ),
    )
    try:
        devices = {str(b.device) for b in compiled.specialized.bindings.values()}
        guard = compiled.specialized.validation.get("baseline_guard") or {}
        assert guard.get("selected") in {"cpu", "streamed", "gpu_prefix_overflow"}
        if guard.get("selected") == "gpu_prefix_overflow":
            assert any(d.startswith("cuda_") for d in devices)
            assert any(d.startswith("cpu") for d in devices)
            assert len(compiled._program.regions) > 1  # noqa: SLF001
            assert compiled.specialized.validation.get("gpu_prefix_cpu_overflow") is True
        elif all(d.startswith("cpu") for d in devices):
            assert len(compiled._program.regions) == 1  # noqa: SLF001
            assert compiled.specialized.validation.get("fused_cpu_baseline") is True
        else:
            assert any(d.startswith("cuda_") for d in devices)
            assert len(compiled._program.regions) > 1
    finally:
        compiled.close()


def test_eager_fused_module_selected_for_beyond_vram() -> None:
    """Beyond-VRAM compile must install an eager-fused DirectPlan (correctness)."""
    import pytest
    from benchmarks.suites.workloads import DeepMLP

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    model = DeepMLP(2048, 48).eval()
    x = torch.randn(4, 2048)
    compiled = tt.compile(
        model,
        example_inputs=(x,),
        config=tt.CompileConfig(
            use_torch_compile=False,
            measure_regions=False,
            allow_gpu=True,
            allow_cpu=True,
            vram_budget_bytes=32 << 20,
            max_region_nodes=8,
            online_profile_feedback=False,
        ),
    )
    try:
        assert compiled.specialized.validation.get("fused_cpu_baseline") is True
        assert compiled.specialized.validation.get("eager_fused_module") is True
        assert compiled.executor.direct_plan is not None
        assert "eager fused" in compiled.executor.direct_plan.reason
        with torch.inference_mode():
            err = (compiled(x) - model(x)).abs().max().item()
        assert err < 1e-5
    finally:
        compiled.close()


@pytest.mark.hardware
def test_eager_fused_module_matches_eager_throughput() -> None:
    """Export-free fused DirectPlan stays near eager wall time (isolated timing)."""
    import pytest
    from benchmarks.suites.workloads import DeepMLP

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    model = DeepMLP(2048, 48).eval()
    x = torch.randn(4, 2048)
    prev_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.inference_mode():
            for _ in range(3):
                model(x)
            pre_samples: list[float] = []
            for _ in range(9):
                t0 = time.perf_counter()
                model(x)
                pre_samples.append(time.perf_counter() - t0)
            pre_samples.sort()
            pre_s = pre_samples[len(pre_samples) // 2]

        compiled = tt.compile(
            model,
            example_inputs=(x,),
            config=tt.CompileConfig(
                use_torch_compile=False,
                measure_regions=False,
                allow_gpu=True,
                allow_cpu=True,
                vram_budget_bytes=32 << 20,
                max_region_nodes=8,
                online_profile_feedback=False,
            ),
        )
        try:
            assert compiled.specialized.validation.get("eager_fused_module") is True
            export_free = bool(compiled.specialized.validation.get("eager_fused_export_free"))
            with torch.inference_mode():
                for _ in range(5):
                    compiled(x)
                tt_samples: list[float] = []
                for _ in range(11):
                    t0 = time.perf_counter()
                    compiled(x)
                    tt_samples.append(time.perf_counter() - t0)
                tt_samples.sort()
                tt_s = tt_samples[len(tt_samples) // 2]
            if export_free:
                assert tt_s / pre_s < 1.35, (tt_s, pre_s, "export_free")
            else:
                with torch.inference_mode():
                    eager_samples: list[float] = []
                    for _ in range(11):
                        t0 = time.perf_counter()
                        model(x)
                        eager_samples.append(time.perf_counter() - t0)
                    eager_samples.sort()
                    eager_s = eager_samples[len(eager_samples) // 2]
                assert tt_s / eager_s < 1.5, (tt_s, eager_s)
        finally:
            compiled.close()
    finally:
        torch.set_num_threads(prev_threads)


def test_auto_bakeoff_runs_when_planner_picks_gpu() -> None:
    """DeepMLP that fits VRAM: planner may pick GPU; bakeoff must still measure CPU."""
    import pytest
    from benchmarks.suites.workloads import DeepMLP

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    model = DeepMLP(256, 6).eval()
    x = torch.randn(2, 256)
    compiled = tt.compile(
        model,
        example_inputs=(x,),
        config=tt.CompileConfig(use_torch_compile=False, measure_regions=False),
    )
    try:
        guard = compiled.specialized.validation.get("baseline_guard") or {}
        assert guard.get("measured") is True
        assert guard.get("selected") in {"cpu", "accelerator"}
        assert "cpu_fused_s" in guard
        assert "accelerator_fused_s" in guard
        devices = {str(b.device) for b in compiled.specialized.bindings.values()}
        if guard["selected"] == "cpu":
            assert all(d.startswith("cpu") for d in devices)
            assert compiled.specialized.validation.get("fused_cpu_baseline") is True
        else:
            assert any(d.startswith("cuda_") for d in devices)
    finally:
        compiled.close()


def test_beyond_vram_bakeoff_records_gpu_prefix_overflow_arm() -> None:
    """Interior hoist cut → bakeoff records (and may select) GPU-prefix overflow."""
    from benchmarks.suites.workloads import DeepMLP

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    model = DeepMLP(128, 12).eval()
    x = torch.randn(2, 128)
    compiled = tt.compile(
        model,
        example_inputs=(x,),
        config=tt.CompileConfig(
            use_torch_compile=False,
            measure_regions=False,
            allow_gpu=True,
            allow_cpu=True,
            vram_budget_bytes=80 << 10,
            max_region_nodes=1,
            online_profile_feedback=False,
        ),
    )
    try:
        guard = compiled.specialized.validation.get("baseline_guard") or {}
        assert guard.get("selected") in {"cpu", "streamed", "gpu_prefix_overflow"}
        meta = guard.get("overflow_meta") or {}
        n_regions = int(meta.get("region_count") or 0)
        n_gpu = int(meta.get("gpu_prefix_regions") or 0)
        if n_regions >= 2 and 0 < n_gpu < n_regions:
            reason = str(meta.get("reason") or "")
            assert reason in {"measured", "predicted"} or reason.startswith("specialize_failed")
        if guard.get("selected") == "gpu_prefix_overflow":
            devices = {str(b.device) for b in compiled.specialized.bindings.values()}
            assert any(d.startswith("cuda_") for d in devices)
            assert any(d.startswith("cpu") for d in devices)
            assert compiled.specialized.validation.get("gpu_prefix_cpu_overflow") is True
            with torch.inference_mode():
                err = (compiled(x) - model(x)).abs().max().item()
            assert err < 1e-4
    finally:
        compiled.close()
