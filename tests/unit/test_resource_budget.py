"""Unit tests for tensortorrent.hardware.budget — pure logic against fake cgroup trees."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tensortorrent.hardware import budget as _b

_MiB = 1 << 20
_GiB = 1 << 30
_256_MiB = 256 * _MiB
_128_MiB = 128 * _MiB
_2_GiB = 2 * _GiB
_768_MiB = 768 * _MiB


# ---------------------------------------------------------------------------
# Helpers to build fake cgroup trees
# ---------------------------------------------------------------------------


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _make_v2(root: Path, *, max_val: str = "max", high_val: str = "max", current: int = 0) -> str:
    _write(root / "memory.max", max_val)
    _write(root / "memory.high", high_val)
    _write(root / "memory.current", str(current))
    return str(root)


def _make_v1(root: Path, *, limit: int, usage: int = 0) -> str:
    _write(root / "memory" / "memory.limit_in_bytes", str(limit))
    _write(root / "memory" / "memory.usage_in_bytes", str(usage))
    return str(root)


# ---------------------------------------------------------------------------
# cgroup v2 memory.max / memory.high / memory.current
# ---------------------------------------------------------------------------


def test_v2_max_sentinel_returns_none(tmp_path: Path) -> None:
    """When both limits are 'max', cgroup_v2 returns None (unlimited)."""
    root = _make_v2(tmp_path, max_val="max", high_val="max")
    avail, detail = _b._read_cgroup_v2_memory(root)
    assert avail is None
    assert "unlimited" in detail


def test_v2_max_limit_used_when_high_is_sentinel(tmp_path: Path) -> None:
    ceiling = 2 * _GiB
    current = 512 * _MiB
    root = _make_v2(tmp_path, max_val=str(ceiling), high_val="max", current=current)
    avail, detail = _b._read_cgroup_v2_memory(root)
    assert avail == ceiling - current
    assert "cgroup_v2" in detail


def test_v2_high_limit_used_when_max_is_sentinel(tmp_path: Path) -> None:
    high = 1 * _GiB
    current = 200 * _MiB
    root = _make_v2(tmp_path, max_val="max", high_val=str(high), current=current)
    avail, _ = _b._read_cgroup_v2_memory(root)
    assert avail == high - current


def test_v2_min_of_max_and_high_used(tmp_path: Path) -> None:
    """min(max, high) is the ceiling; current is subtracted."""
    ceiling_max = 4 * _GiB
    ceiling_high = 1 * _GiB
    current = 100 * _MiB
    root = _make_v2(tmp_path, max_val=str(ceiling_max), high_val=str(ceiling_high), current=current)
    avail, _ = _b._read_cgroup_v2_memory(root)
    assert avail == ceiling_high - current


def test_v2_never_returns_negative(tmp_path: Path) -> None:
    """current > ceiling (over-committed): available is clamped to 0."""
    root = _make_v2(tmp_path, max_val=str(100 * _MiB), high_val="max", current=200 * _MiB)
    avail, _ = _b._read_cgroup_v2_memory(root)
    assert avail == 0


def test_v2_not_present_returns_none(tmp_path: Path) -> None:
    """No memory.max or memory.high → returns None."""
    avail, reason = _b._read_cgroup_v2_memory(str(tmp_path))
    assert avail is None
    assert "not present" in reason


# ---------------------------------------------------------------------------
# cgroup v1 limit/usage + unlimited sentinel
# ---------------------------------------------------------------------------


def test_v1_basic_available(tmp_path: Path) -> None:
    limit = 2 * _GiB
    usage = 500 * _MiB
    root = _make_v1(tmp_path, limit=limit, usage=usage)
    avail, detail = _b._read_cgroup_v1_memory(root)
    assert avail == limit - usage
    assert "cgroup_v1" in detail


def test_v1_unlimited_sentinel_returns_none(tmp_path: Path) -> None:
    """Values >= 0x7FFFFFFFFFFFF000 indicate unlimited in cgroup v1."""
    root = _make_v1(tmp_path, limit=0x7FFFFFFFFFFFF000)
    avail, detail = _b._read_cgroup_v1_memory(root)
    assert avail is None
    assert "unlimited" in detail


def test_v1_not_present_returns_none(tmp_path: Path) -> None:
    avail, reason = _b._read_cgroup_v1_memory(str(tmp_path))
    assert avail is None
    assert "not present" in reason


# ---------------------------------------------------------------------------
# min(cgroup, os_available) selection
# ---------------------------------------------------------------------------


def test_resolve_host_prefers_cgroup_v2_over_os(tmp_path: Path, monkeypatch: Any) -> None:
    """When cgroup_v2 is smaller than os_available, cgroup_v2 wins."""
    small = 256 * _MiB
    _make_v2(tmp_path, max_val=str(small), high_val="max", current=0)

    # Fake psutil to report a huge OS available
    class FakeVM:
        available = 8 * _GiB
        total = 16 * _GiB

    monkeypatch.setattr(_b, "__name__", _b.__name__)
    import psutil

    monkeypatch.setattr(psutil, "virtual_memory", lambda: FakeVM())

    result = _b.resolve_host_memory_budget(cgroup_root=str(tmp_path))
    assert result.source.kind == "cgroup_v2"
    # raw = small; reserve = clamp(5%, [256MiB, 2GiB]); allowed = max(128MiB, raw-reserve)
    assert result.allowed_bytes >= _128_MiB


def test_resolve_host_prefers_os_when_cgroup_larger(tmp_path: Path, monkeypatch: Any) -> None:
    """When os_available < cgroup, os_available (min) wins."""
    huge = 16 * _GiB
    _make_v2(tmp_path, max_val=str(huge), high_val="max", current=0)

    class FakeVM:
        available = 512 * _MiB
        total = 8 * _GiB

    import psutil

    monkeypatch.setattr(psutil, "virtual_memory", lambda: FakeVM())
    result = _b.resolve_host_memory_budget(cgroup_root=str(tmp_path))
    assert result.source.kind == "os_available"


# ---------------------------------------------------------------------------
# Explicit override wins unconditionally
# ---------------------------------------------------------------------------


def test_explicit_override_wins(tmp_path: Path, monkeypatch: Any) -> None:
    """Explicit value wins even when cgroup and os say otherwise."""
    _make_v2(tmp_path, max_val=str(128 * _MiB), high_val="max", current=0)

    class FakeVM:
        available = 256 * _MiB
        total = 1 * _GiB

    import psutil

    monkeypatch.setattr(psutil, "virtual_memory", lambda: FakeVM())

    explicit = 4 * _GiB
    result = _b.resolve_host_memory_budget(explicit=explicit, cgroup_root=str(tmp_path))
    assert result.source.kind == "explicit"
    # allowed = max(128MiB, explicit - reserve)
    reserve = _b._default_reserve(explicit)
    assert result.allowed_bytes == max(_128_MiB, explicit - reserve)


# ---------------------------------------------------------------------------
# reserve clamp [256MiB, 2GiB] and TT_HOST_MEMORY_RESERVE_BYTES override
# ---------------------------------------------------------------------------


def test_default_reserve_clamp_lower_bound(monkeypatch: Any) -> None:
    """5% of small raw is below 256 MiB → clamped to 256 MiB."""
    monkeypatch.delenv("TT_HOST_MEMORY_RESERVE_BYTES", raising=False)
    raw = 1 * _GiB  # 5% = 51.2 MiB < 256 MiB
    reserve = _b._default_reserve(raw)
    assert reserve == _256_MiB


def test_default_reserve_clamp_upper_bound(monkeypatch: Any) -> None:
    """5% of huge raw is above 2 GiB → clamped to 2 GiB."""
    monkeypatch.delenv("TT_HOST_MEMORY_RESERVE_BYTES", raising=False)
    raw = 200 * _GiB  # 5% = 10 GiB >> 2 GiB
    reserve = _b._default_reserve(raw)
    assert reserve == _2_GiB


def test_reserve_env_override(monkeypatch: Any) -> None:
    """TT_HOST_MEMORY_RESERVE_BYTES overrides the default clamp."""
    monkeypatch.setenv("TT_HOST_MEMORY_RESERVE_BYTES", str(100 * _MiB))
    reserve = _b._default_reserve(10 * _GiB)
    assert reserve == 100 * _MiB


def test_allowed_floor_128_mib(tmp_path: Path, monkeypatch: Any) -> None:
    """allowed is never below 128 MiB even when raw - reserve would be less."""
    # Set a large reserve and a tiny budget
    monkeypatch.setenv("TT_HOST_MEMORY_RESERVE_BYTES", str(200 * _MiB))

    class FakeVM:
        available = 100 * _MiB  # tiny
        total = 8 * _GiB

    import psutil

    monkeypatch.setattr(psutil, "virtual_memory", lambda: FakeVM())
    result = _b.resolve_host_memory_budget(cgroup_root=str(tmp_path))
    assert result.allowed_bytes >= _128_MiB


# ---------------------------------------------------------------------------
# resolve_cpu_budget
# ---------------------------------------------------------------------------


def test_resolve_cpu_explicit(tmp_path: Path) -> None:
    count, source = _b.resolve_cpu_budget(explicit=3, cgroup_root=str(tmp_path))
    assert count == 3
    assert source.kind == "explicit"


def test_resolve_cpu_affinity_used(tmp_path: Path, monkeypatch: Any) -> None:
    """When os.sched_getaffinity available, it should contribute to candidates."""
    import os

    monkeypatch.setattr(os, "sched_getaffinity", lambda _: {0, 1}, raising=False)
    # Also ensure cpu_count returns same or larger
    monkeypatch.setattr(os, "cpu_count", lambda: 8)
    count, _ = _b.resolve_cpu_budget(cgroup_root=str(tmp_path))
    # Min of (2 affinity, 8 cpu_count) = 2
    assert count == 2


# ---------------------------------------------------------------------------
# resolve_device_memory_budget
# ---------------------------------------------------------------------------


def test_device_explicit_path() -> None:
    result = _b.resolve_device_memory_budget(
        total_bytes=8 * _GiB, free_bytes=4 * _GiB, explicit=2 * _GiB, headroom_bytes=_256_MiB
    )
    assert result.source.kind == "explicit"
    assert result.allowed_bytes == 2 * _GiB


def test_device_free_path() -> None:
    """With free_bytes present: allowed = free - headroom."""
    free = 6 * _GiB
    headroom = _768_MiB
    result = _b.resolve_device_memory_budget(
        total_bytes=8 * _GiB, free_bytes=free, explicit=None, headroom_bytes=headroom
    )
    assert result.source.kind == "os_available"
    assert result.allowed_bytes == free - headroom


def test_device_total_fallback(monkeypatch: Any) -> None:
    """Without free_bytes: total*0.9 - headroom."""
    total = 8 * _GiB
    headroom = _256_MiB
    result = _b.resolve_device_memory_budget(total_bytes=total, free_bytes=None, explicit=None, headroom_bytes=headroom)
    assert result.source.kind == "total_fallback"
    expected = max(0, int(total * 0.9) - headroom)
    assert result.allowed_bytes == expected
    assert "live free memory was unavailable" in result.notes[0]


def test_device_never_negative() -> None:
    """allowed is clamped to 0."""
    result = _b.resolve_device_memory_budget(
        total_bytes=100 * _MiB, free_bytes=10 * _MiB, explicit=None, headroom_bytes=500 * _MiB
    )
    assert result.allowed_bytes == 0


# ---------------------------------------------------------------------------
# default_vram_headroom_bytes
# ---------------------------------------------------------------------------


def test_vram_headroom_display(monkeypatch: Any) -> None:
    monkeypatch.delenv("TT_VRAM_HEADROOM_BYTES", raising=False)
    assert _b.default_vram_headroom_bytes(display_active=True) == _768_MiB


def test_vram_headroom_headless(monkeypatch: Any) -> None:
    monkeypatch.delenv("TT_VRAM_HEADROOM_BYTES", raising=False)
    assert _b.default_vram_headroom_bytes(display_active=False) == _256_MiB


def test_vram_headroom_env_override(monkeypatch: Any) -> None:
    monkeypatch.setenv("TT_VRAM_HEADROOM_BYTES", str(512 * _MiB))
    assert _b.default_vram_headroom_bytes(display_active=True) == 512 * _MiB
    assert _b.default_vram_headroom_bytes(display_active=False) == 512 * _MiB


# ---------------------------------------------------------------------------
# vram_capacity_floor_bytes
# ---------------------------------------------------------------------------


def test_vram_capacity_floor_basic(monkeypatch: Any) -> None:
    monkeypatch.delenv("TT_DISABLE_VRAM_CAPACITY_FLOOR", raising=False)
    assert _b.vram_capacity_floor_bytes(8 * _GiB, _768_MiB) == 8 * _GiB - _768_MiB


def test_vram_capacity_floor_never_negative(monkeypatch: Any) -> None:
    monkeypatch.delenv("TT_DISABLE_VRAM_CAPACITY_FLOOR", raising=False)
    assert _b.vram_capacity_floor_bytes(100 * _MiB, 500 * _MiB) == 0


def test_vram_capacity_floor_disabled(monkeypatch: Any) -> None:
    monkeypatch.setenv("TT_DISABLE_VRAM_CAPACITY_FLOOR", "1")
    assert _b.vram_capacity_floor_bytes(8 * _GiB, _768_MiB) == 0


# ---------------------------------------------------------------------------
# resolve_disk_budget — 80%
# ---------------------------------------------------------------------------


def test_disk_budget_80_percent(tmp_path: Path) -> None:
    result = _b.resolve_disk_budget(tmp_path)
    import shutil

    usage = shutil.disk_usage(str(tmp_path))
    expected = int(usage.free * 0.8)
    assert result.allowed_bytes == expected
    assert result.source.kind == "os_available"


def test_disk_budget_explicit(tmp_path: Path) -> None:
    explicit = 1 * _GiB
    result = _b.resolve_disk_budget(tmp_path, explicit=explicit)
    assert result.allowed_bytes == explicit
    assert result.source.kind == "explicit"
