"""Tests for platform guard logic in CompileConfig (process_workers, WSL2, presets)."""

from __future__ import annotations

import sys
import warnings
from typing import Any

import pytest

from tensortorrent.config import CompileConfig
from tensortorrent.errors import ConfigurationError

# ---------------------------------------------------------------------------
# process_workers on non-Linux raises ConfigurationError
# ---------------------------------------------------------------------------


def test_process_workers_on_darwin_raises(monkeypatch: Any) -> None:
    """process_workers > 0 on macOS must raise ConfigurationError."""
    monkeypatch.setattr(sys, "platform", "darwin")
    with pytest.raises(ConfigurationError, match="process_workers"):
        CompileConfig(process_workers=2)


def test_process_workers_on_win_raises(monkeypatch: Any) -> None:
    """process_workers > 0 on Windows must raise ConfigurationError."""
    monkeypatch.setattr(sys, "platform", "win32")
    with pytest.raises(ConfigurationError, match="process_workers"):
        CompileConfig(process_workers=1)


def test_process_workers_zero_is_fine_on_any_platform(monkeypatch: Any) -> None:
    """process_workers=0 must be accepted on all platforms."""
    monkeypatch.setattr(sys, "platform", "darwin")
    cfg = CompileConfig(process_workers=0)
    assert cfg.process_workers == 0


# ---------------------------------------------------------------------------
# WSL2 + process_workers > 0 → warnings.warn
# ---------------------------------------------------------------------------


def test_process_workers_on_wsl2_emits_warning(monkeypatch: Any) -> None:
    """process_workers > 0 under WSL2 must emit a warning (not raise)."""
    monkeypatch.setattr(sys, "platform", "linux")
    import tensortorrent.hardware.budget as budget_mod

    monkeypatch.setattr(budget_mod, "is_wsl2", lambda: True)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cfg = CompileConfig(process_workers=1)
    assert cfg.process_workers == 1
    assert any("WSL2" in str(warning.message) or "wsl" in str(warning.message).lower() for warning in w), (
        f"expected WSL2 warning, got: {[str(x.message) for x in w]}"
    )


# ---------------------------------------------------------------------------
# polite() preset values
# ---------------------------------------------------------------------------


def test_polite_preset_values() -> None:
    """CompileConfig.polite() must match the documented preset values."""
    _1_5_GiB = int(1.5 * (1 << 30))
    cfg = CompileConfig.polite()
    assert cfg.vram_headroom_bytes == _1_5_GiB
    assert cfg.stall_timeout_s == 120.0
    assert cfg.max_concurrent_regions == 1
    assert cfg.prefetch_distance == 1


# ---------------------------------------------------------------------------
# New config field validation: negative/zero/bool rejection
# ---------------------------------------------------------------------------


def test_negative_max_plan_candidates_rejected() -> None:
    with pytest.raises(ValueError, match="max_plan_candidates"):
        CompileConfig(max_plan_candidates=0)


def test_negative_region_nodes_rejected() -> None:
    with pytest.raises(ValueError, match="max_region_nodes"):
        CompileConfig(max_region_nodes=-1)


def test_bool_passed_as_int_field_rejected() -> None:
    """Booleans must not be accepted where an int is required."""
    with pytest.raises(TypeError):
        CompileConfig(max_plan_candidates=True)  # type: ignore[arg-type]


def test_negative_stall_timeout_rejected() -> None:
    with pytest.raises(ValueError, match="stall_timeout_s"):
        CompileConfig(stall_timeout_s=-1.0)


def test_zero_ram_budget_rejected() -> None:
    with pytest.raises(ValueError, match="ram_budget_bytes"):
        CompileConfig(ram_budget_bytes=0)


def test_bool_as_budget_rejected() -> None:
    with pytest.raises(TypeError):
        CompileConfig(ram_budget_bytes=True)  # type: ignore[arg-type]


def test_negative_vram_headroom_bytes_rejected() -> None:
    with pytest.raises(ValueError, match="vram_headroom_bytes"):
        CompileConfig(vram_headroom_bytes=0)


def test_bool_as_allow_cpu_rejected() -> None:
    """Bool for a bool field is fine — but check that non-bool raises."""
    with pytest.raises(TypeError):
        CompileConfig(allow_cpu=1)  # type: ignore[arg-type]


def test_negative_prefetch_distance_rejected() -> None:
    with pytest.raises(ValueError, match="prefetch_distance"):
        CompileConfig(prefetch_distance=-1)


# ---------------------------------------------------------------------------
# from_json_dict unknown-key warning captured via caplog
# ---------------------------------------------------------------------------


def test_from_json_dict_unknown_key_warning(caplog: Any) -> None:
    """from_json_dict with an unknown key must log a warning."""
    import logging

    data = CompileConfig().to_json_dict()
    data["totally_unknown_key_xyz"] = "oops"

    with caplog.at_level(logging.WARNING, logger="tensortorrent.config"):
        cfg = CompileConfig.from_json_dict(data)

    assert any("totally_unknown_key_xyz" in rec.message for rec in caplog.records), (
        f"expected warning about unknown key, got: {[r.message for r in caplog.records]}"
    )
    assert isinstance(cfg, CompileConfig)


def test_from_json_dict_roundtrip_clean() -> None:
    """A config survives to_json_dict → from_json_dict preserving values."""
    cfg = CompileConfig(
        max_concurrent_regions=2,
        prefetch_distance=2,
        stall_timeout_s=60.0,
    )
    data = cfg.to_json_dict()
    restored = CompileConfig.from_json_dict(data)
    assert restored.max_concurrent_regions == 2
    assert restored.prefetch_distance == 2
