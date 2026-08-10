"""Regression coverage for export-free fused CPU state and capacity semantics."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn

import tensortorrent.runtime.capacity as capacity_mod
from tensortorrent.compile.eager_cpu import (
    build_eager_fused_compiled_module,
    build_eager_fused_program,
    module_parameter_bytes,
)
from tensortorrent.config import CompileConfig
from tensortorrent.errors import MemoryCapacityError
from tensortorrent.runtime.capacity import (
    CapacityBudgets,
    _resolve_capacity_footprint,
    build_module_capacity_ledger,
)


def _model() -> nn.Module:
    return nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 4)).eval()


def _build_compiled(model: nn.Module, x: torch.Tensor) -> Any:
    state_bytes = module_parameter_bytes(model)
    config = CompileConfig(
        allow_cpu=True,
        allow_gpu=False,
        use_torch_compile=False,
        measure_regions=False,
        validate_numerics=False,
        host_memory_reserve_bytes=4096,
    )
    return build_eager_fused_compiled_module(
        model,
        (x,),
        config=config,
        name="export_free_contract",
        guard={
            "cpu_fused_s": 0.001,
            "streamed_predicted_s": 1.0,
            "param_bytes": state_bytes,
            "selected": "cpu",
        },
    )


def test_export_free_total_state_bytes_counts_resident_root() -> None:
    model = _model()
    x = torch.randn(2, 16)
    program = build_eager_fused_program(model, (x,), name="state_bytes")
    assert program.state_bindings == {}
    assert program.total_state_bytes() == module_parameter_bytes(model)
    assert program.max_region_state_bytes() == module_parameter_bytes(model)


def test_export_free_explicit_ram_budget_reserves_resident_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _model()
    x = torch.randn(2, 16)
    state_bytes = module_parameter_bytes(model)
    reserve = 4096
    # Explicit ceiling after reserve. Monkeypatch around the 128 MiB host floor so
    # the ledger math under test is the export-free base reservation itself.
    allowed_after_reserve = state_bytes + 8192
    program = build_eager_fused_program(model, (x,), name="capacity")
    plan = SimpleNamespace(devices_used=("cpu_numa_0",), predicted_peak_bytes={})
    store = SimpleNamespace(needs_prefetch=False, kind="eager_fused")
    monkeypatch.setattr(
        capacity_mod,
        "resolve_capacity_budgets",
        lambda config, machine=None: CapacityBudgets(
            host_bytes=allowed_after_reserve,
            device_bytes=0,
            disk_bytes=0,
            host_source_kind="explicit",
            host_reflects_live_remaining=False,
        ),
    )

    config = CompileConfig(
        allow_cpu=True,
        allow_gpu=False,
        allow_nvme_streaming=True,
        ram_budget_bytes=allowed_after_reserve + reserve,
        host_memory_reserve_bytes=reserve,
        use_torch_compile=False,
        measure_regions=False,
        validate_numerics=False,
    )
    ledger = build_module_capacity_ledger(
        program=program,
        plan=plan,
        config=config,
        parameter_store=store,
        machine=None,
    )
    assert ledger.budgets.host_bytes == allowed_after_reserve - state_bytes
    assert ledger.per_request.host_bytes >= 1


def test_export_free_explicit_ram_budget_fails_closed_when_weights_do_not_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model()
    x = torch.randn(2, 16)
    state_bytes = module_parameter_bytes(model)
    reserve = 4096
    # After subtracting the required positive reserve, the explicit budget cannot
    # hold the resident export-free weights.
    allowed_after_reserve = max(0, state_bytes - 1)
    program = build_eager_fused_program(model, (x,), name="capacity_fail")
    plan = SimpleNamespace(devices_used=("cpu_numa_0",), predicted_peak_bytes={})
    store = SimpleNamespace(needs_prefetch=False, kind="eager_fused")
    monkeypatch.setattr(
        capacity_mod,
        "resolve_capacity_budgets",
        lambda config, machine=None: CapacityBudgets(
            host_bytes=allowed_after_reserve,
            device_bytes=0,
            disk_bytes=0,
            host_source_kind="explicit",
            host_reflects_live_remaining=False,
        ),
    )

    config = CompileConfig(
        allow_cpu=True,
        allow_gpu=False,
        allow_nvme_streaming=True,
        ram_budget_bytes=allowed_after_reserve + reserve,
        host_memory_reserve_bytes=reserve,
        use_torch_compile=False,
        measure_regions=False,
        validate_numerics=False,
    )
    with pytest.raises(MemoryCapacityError, match="shared host reservation exceeds capacity"):
        build_module_capacity_ledger(
            program=program,
            plan=plan,
            config=config,
            parameter_store=store,
            machine=None,
        )


def test_export_free_tight_ram_budget_does_not_mark_nvme_streaming() -> None:
    model = _model()
    x = torch.randn(2, 16)
    state_bytes = module_parameter_bytes(model)
    program = build_eager_fused_program(model, (x,), name="no_stream")
    plan = SimpleNamespace(devices_used=("cpu_numa_0",), predicted_peak_bytes={})
    store = SimpleNamespace(needs_prefetch=False, kind="eager_fused")
    config = CompileConfig(
        allow_cpu=True,
        allow_gpu=False,
        allow_nvme_streaming=True,
        ram_budget_bytes=1,
        host_memory_reserve_bytes=1024,
        use_torch_compile=False,
        measure_regions=False,
        validate_numerics=False,
    )
    state, streaming, cpu_only, device_base = _resolve_capacity_footprint(
        program=program,
        plan=plan,
        config=config,
        parameter_store=store,
        machine=None,
    )
    assert state == state_bytes
    assert streaming is False
    assert cpu_only is True
    assert device_base == 0


def test_export_free_live_host_budget_skips_double_count(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _model()
    x = torch.randn(2, 16)
    state_bytes = module_parameter_bytes(model)
    program = build_eager_fused_program(model, (x,), name="live_host")
    plan = SimpleNamespace(devices_used=("cpu_numa_0",), predicted_peak_bytes={})
    store = SimpleNamespace(needs_prefetch=False, kind="eager_fused")
    live_allowed = 64 << 20
    monkeypatch.setattr(
        capacity_mod,
        "resolve_capacity_budgets",
        lambda config, machine=None: CapacityBudgets(
            host_bytes=live_allowed,
            device_bytes=0,
            disk_bytes=0,
            host_source_kind="os_available",
            host_reflects_live_remaining=True,
        ),
    )
    config = CompileConfig(
        allow_cpu=True,
        allow_gpu=False,
        host_memory_reserve_bytes=4096,
        use_torch_compile=False,
        measure_regions=False,
        validate_numerics=False,
    )
    ledger = build_module_capacity_ledger(
        program=program,
        plan=plan,
        config=config,
        parameter_store=store,
        machine=None,
    )
    assert ledger.budgets.host_bytes == live_allowed
    assert state_bytes > 0


def test_export_free_state_dict_round_trip_uses_compiled_key_space() -> None:
    model = _model()
    x = torch.randn(2, 16)
    compiled = _build_compiled(model, x)
    try:
        before = compiled(x).detach().clone()
        state = compiled.state_dict()
        eager_state = model.state_dict()
        assert set(state) == {f"graph_module.{key}" for key in eager_state}
        for key, value in eager_state.items():
            torch.testing.assert_close(state[f"graph_module.{key}"], value)

        changed = {key: value.clone() for key, value in state.items()}
        weight_key = next(key for key, value in changed.items() if value.is_floating_point())
        changed[weight_key] = changed[weight_key] + 0.25
        incompatible = compiled.load_state_dict(changed)
        assert incompatible.missing_keys == []
        assert incompatible.unexpected_keys == []
        eager_key = weight_key.removeprefix("graph_module.")
        torch.testing.assert_close(model.state_dict()[eager_key], changed[weight_key])

        after = compiled(x)
        assert not torch.allclose(before, after)
        torch.testing.assert_close(after, model(x))
    finally:
        compiled.close()


def test_export_free_load_state_dict_strictness() -> None:
    model = _model()
    x = torch.randn(2, 16)
    compiled = _build_compiled(model, x)
    try:
        state = compiled.state_dict()
        with pytest.raises(RuntimeError, match="Unexpected key"):
            compiled.load_state_dict({**state, "graph_module.not_a_real_weight": torch.zeros(1)}, strict=True)
        result = compiled.load_state_dict(
            {**state, "graph_module.not_a_real_weight": torch.zeros(1)},
            strict=False,
        )
        assert any("not_a_real_weight" in str(k) for k in result.unexpected_keys)
        assert result.missing_keys == []
    finally:
        compiled.close()


def test_export_free_preserves_caller_train_eval_mode() -> None:
    model = _model()
    model.train()
    assert model.training is True
    x = torch.randn(2, 16)
    compiled = _build_compiled(model, x)
    try:
        assert model.training is True
        assert list(compiled.children()) == []
        compiled.eval()
        assert model.training is True
        assert compiled.training is False
        # Calling the compiled module must not flip the caller's mode either.
        _ = compiled(x)
        assert model.training is True
    finally:
        compiled.close()
