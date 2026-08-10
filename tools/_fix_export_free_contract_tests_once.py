from pathlib import Path

Path("tests/unit/test_export_free_contract.py").write_text('''"""Regression coverage for export-free fused CPU state and capacity semantics."""

from __future__ import annotations

from types import SimpleNamespace

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
from tensortorrent.runtime.capacity import CapacityBudgets, build_module_capacity_ledger


def _model() -> nn.Module:
    return nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 4)).eval()


def _capacity_config() -> CompileConfig:
    # Tiny explicit RAM budget deliberately says the model would need streaming.
    # The export-free executor has a real resident store, so capacity accounting
    # must trust the store rather than pretending this path can stream weights.
    return CompileConfig(
        allow_cpu=True,
        allow_gpu=False,
        allow_nvme_streaming=True,
        ram_budget_bytes=1,
        use_torch_compile=False,
        measure_regions=False,
        validate_numerics=False,
    )


def test_export_free_explicit_ram_budget_reserves_resident_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _model()
    x = torch.randn(2, 16)
    state_bytes = module_parameter_bytes(model)
    program = build_eager_fused_program(model, (x,), name="capacity")
    plan = SimpleNamespace(devices_used=("cpu_numa_0",), predicted_peak_bytes={})
    store = SimpleNamespace(needs_prefetch=False)
    monkeypatch.setattr(
        capacity_mod,
        "resolve_capacity_budgets",
        lambda config, machine=None: CapacityBudgets(
            host_bytes=state_bytes + 4096,
            device_bytes=0,
            disk_bytes=0,
            host_source_kind="explicit",
            host_reflects_live_remaining=False,
        ),
    )

    ledger = build_module_capacity_ledger(
        program=program,
        plan=plan,
        config=_capacity_config(),
        parameter_store=store,
        machine=None,
    )
    assert ledger.budgets.host_bytes == 4096
    assert ledger.per_request.host_bytes >= 1


def test_export_free_explicit_ram_budget_fails_closed_when_weights_do_not_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model()
    x = torch.randn(2, 16)
    state_bytes = module_parameter_bytes(model)
    program = build_eager_fused_program(model, (x,), name="capacity")
    plan = SimpleNamespace(devices_used=("cpu_numa_0",), predicted_peak_bytes={})
    store = SimpleNamespace(needs_prefetch=False)
    monkeypatch.setattr(
        capacity_mod,
        "resolve_capacity_budgets",
        lambda config, machine=None: CapacityBudgets(
            host_bytes=max(0, state_bytes - 1),
            device_bytes=0,
            disk_bytes=0,
            host_source_kind="explicit",
            host_reflects_live_remaining=False,
        ),
    )

    with pytest.raises(MemoryCapacityError, match="shared host reservation exceeds capacity"):
        build_module_capacity_ledger(
            program=program,
            plan=plan,
            config=_capacity_config(),
            parameter_store=store,
            machine=None,
        )


def test_export_free_state_dict_round_trip_uses_compiled_key_space() -> None:
    model = _model()
    x = torch.randn(2, 16)
    state_bytes = module_parameter_bytes(model)
    config = CompileConfig(
        allow_cpu=True,
        allow_gpu=False,
        use_torch_compile=False,
        measure_regions=False,
        validate_numerics=False,
    )
    compiled = build_eager_fused_compiled_module(
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
    try:
        state = compiled.state_dict()
        eager_state = model.state_dict()
        assert set(state) == {f"graph_module.{key}" for key in eager_state}
        for key, value in eager_state.items():
            torch.testing.assert_close(state[f"graph_module.{key}"], value)

        changed = state.copy()
        weight_key = next(key for key, value in changed.items() if value.is_floating_point())
        changed[weight_key] = changed[weight_key].clone() + 0.25
        incompatible = compiled.load_state_dict(changed)
        assert incompatible.missing_keys == []
        assert incompatible.unexpected_keys == []
        eager_key = weight_key.removeprefix("graph_module.")
        torch.testing.assert_close(model.state_dict()[eager_key], changed[weight_key])
    finally:
        compiled.close()
''')
