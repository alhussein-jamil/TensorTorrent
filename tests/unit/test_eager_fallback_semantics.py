"""Semantic regressions for eager fused DirectPlan / bakeoff fallback."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

import tensortorrent as tt


class _TrainyNet(nn.Module):
    """Module with Dropout + BatchNorm that differs in train vs eval."""

    def __init__(self, width: int = 64) -> None:
        super().__init__()
        self.fc1 = nn.Linear(width, width)
        self.bn = nn.BatchNorm1d(width)
        self.drop = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(width, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.bn(x)
        x = self.drop(x)
        return self.fc2(x)


class _KwargNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(16, 8)

    def forward(self, x: torch.Tensor, *, scale: float = 1.0) -> torch.Tensor:
        return self.fc(x) * scale


class _NestedOutNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(16, 8)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        y = self.fc(x)
        return y, {"aux": y * 2}


def test_train_mode_module_compiles_with_eval_semantics() -> None:
    """Capture forces eval; eager fused path must keep Dropout/BN eval behavior."""
    torch.manual_seed(0)
    model = _TrainyNet().train()  # intentionally train()
    x = torch.randn(4, 64)
    with torch.inference_mode():
        model.eval()
        expected = model(x).clone()
        model.train()

    compiled = tt.compile(
        model,
        example_inputs=(x,),
        config=tt.CompileConfig(
            use_torch_compile=False,
            measure_regions=False,
            allow_gpu=False,
            allow_cpu=True,
            validate_numerics=False,
        ),
    )
    try:
        # Force the eager fused path when available; otherwise export path still eval.
        if compiled.specialized.validation.get("fused_cpu_baseline"):
            assert compiled.specialized.validation.get("eager_fused_module") is True
        # compile() must restore the caller's train/eval mode (eval only temporarily).
        assert model.training is True
        with torch.inference_mode():
            out = compiled(x)
        torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)
        # Stochastic train-mode Dropout would disagree across calls.
        with torch.inference_mode():
            out2 = compiled(x)
        torch.testing.assert_close(out, out2, atol=0, rtol=0)
    finally:
        compiled.close()


def test_positional_and_kwarg_inputs() -> None:
    model = _KwargNet().eval()
    x = torch.randn(2, 16)
    compiled = tt.compile(
        model,
        example_inputs=((x,), {"scale": 2.0}),
        config=tt.CompileConfig(
            use_torch_compile=False,
            measure_regions=False,
            allow_gpu=False,
            validate_numerics=False,
        ),
    )
    try:
        with torch.inference_mode():
            expected = model(x, scale=2.0)
            out = compiled(x, scale=2.0)
        torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)
    finally:
        compiled.close()


def test_nested_tuple_dict_outputs() -> None:
    model = _NestedOutNet().eval()
    x = torch.randn(2, 16)
    compiled = tt.compile(
        model,
        example_inputs=(x,),
        config=tt.CompileConfig(
            use_torch_compile=False,
            measure_regions=False,
            allow_gpu=False,
            validate_numerics=False,
        ),
    )
    try:
        with torch.inference_mode():
            expected = model(x)
            out = compiled(x)
        assert isinstance(out, tuple) and len(out) == 2
        torch.testing.assert_close(out[0], expected[0], atol=1e-5, rtol=1e-5)
        assert isinstance(out[1], dict)
        torch.testing.assert_close(out[1]["aux"], expected[1]["aux"], atol=1e-5, rtol=1e-5)
    finally:
        compiled.close()


def test_save_reload_without_eager_module(tmp_path: Path) -> None:
    model = _TrainyNet().train()
    x = torch.randn(4, 64)
    artifact = tmp_path / "art"
    compiled = tt.compile(
        model,
        example_inputs=(x,),
        config=tt.CompileConfig(
            use_torch_compile=False,
            measure_regions=False,
            allow_gpu=False,
            validate_numerics=False,
        ),
        artifact_dir=artifact,
    )
    try:
        with torch.inference_mode():
            before = compiled(x).clone()
    finally:
        compiled.close()
        del model

    reloaded = tt.load_compiled(artifact)
    try:
        with torch.inference_mode():
            after = reloaded(x)
        torch.testing.assert_close(after, before, atol=1e-5, rtol=1e-5)
    finally:
        reloaded.close()


def test_cpu_direct_plan_does_not_reserve_gpu_capacity() -> None:
    model = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 8)).eval()
    x = torch.randn(4, 32)
    compiled = tt.compile(
        model,
        example_inputs=(x,),
        config=tt.CompileConfig(
            use_torch_compile=False,
            measure_regions=False,
            allow_gpu=True,
            allow_cpu=True,
            # Tiny VRAM budget → auto should prefer fused CPU when GPU loses.
            vram_budget_bytes=1 << 20,
            validate_numerics=False,
        ),
    )
    try:
        devices = [str(d) for d in compiled.specialized.plan.devices_used]
        if not all(d.startswith("cpu") for d in devices):
            pytest.skip(f"CPU path not selected on this host: {devices}")
        from tensortorrent.runtime.capacity import build_module_capacity_ledger, resolve_capacity_budgets

        machine = getattr(compiled, "_machine", None)
        rebuilt = build_module_capacity_ledger(
            program=compiled._program,  # noqa: SLF001
            plan=compiled.specialized.plan,
            config=compiled.config,
            parameter_store=compiled.executor.parameter_store,
            machine=machine,
        )
        raw = resolve_capacity_budgets(compiled.config, machine=machine)
        # CPU-only plan: no shared VRAM base reservation and no per-request device lease.
        assert rebuilt.per_request.device_bytes == 0
        assert rebuilt.budgets.device_bytes == raw.device_bytes
    finally:
        compiled.close()


def test_parameter_state_dict_updates_outputs() -> None:
    """Compiled weights remain mutable via state_dict; forwards see the new values."""
    model = nn.Sequential(nn.Linear(16, 16, bias=False), nn.ReLU(), nn.Linear(16, 4, bias=False)).eval()
    x = torch.randn(2, 16)
    compiled = tt.compile(
        model,
        example_inputs=(x,),
        config=tt.CompileConfig(
            use_torch_compile=False,
            measure_regions=False,
            allow_gpu=False,
            validate_numerics=False,
        ),
    )
    try:
        with torch.inference_mode():
            before = compiled(x).clone()
        sd = {k: v.clone() for k, v in compiled.state_dict().items()}
        for key, value in sd.items():
            if torch.is_tensor(value) and value.is_floating_point():
                sd[key] = value + 0.5
        compiled.load_state_dict(sd)
        with torch.inference_mode():
            after = compiled(x)
        assert not torch.allclose(before, after, atol=1e-6, rtol=1e-6)
    finally:
        compiled.close()
