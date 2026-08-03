"""Tests for Python-side spill directory safety and sweep logic."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from tensortorrent.errors import ConfigurationError
from tensortorrent.runtime.native_bridge import _check_not_tmpfs, _resolve_spill_dir

# ---------------------------------------------------------------------------
# _check_not_tmpfs
# ---------------------------------------------------------------------------


def test_check_not_tmpfs_raises_for_shm(tmp_path: Path) -> None:
    """/dev/shm is a real tmpfs on Linux — must raise ConfigurationError."""
    shm = Path("/dev/shm")
    if not shm.exists():
        pytest.skip("/dev/shm not present on this platform")
    # Use a subpath to avoid touching real shm files unnecessarily
    test_path = shm / "tt_test_tmpfs_check"
    try:
        test_path.mkdir(exist_ok=True)
        with pytest.raises(ConfigurationError, match="tmpfs"):
            _check_not_tmpfs(test_path)
    finally:
        if test_path.exists():
            test_path.rmdir()


def test_check_not_tmpfs_bypassed_with_env(monkeypatch: Any) -> None:
    """TT_ALLOW_TMPFS_SPILL=1 prevents the ConfigurationError."""
    shm = Path("/dev/shm")
    if not shm.exists():
        pytest.skip("/dev/shm not present on this platform")
    test_path = shm / "tt_test_tmpfs_bypass"
    try:
        test_path.mkdir(exist_ok=True)
        monkeypatch.setenv("TT_ALLOW_TMPFS_SPILL", "1")
        # The check is called inside _resolve_spill_root, not _check_not_tmpfs directly.
        # Directly calling _check_not_tmpfs bypasses the env guard — use _resolve_spill_dir.
        # Per implementation: _resolve_spill_root checks env BEFORE calling _check_not_tmpfs.
        from tensortorrent.runtime.native_bridge import _resolve_spill_root

        result = _resolve_spill_root(str(test_path), None)
        # Should not raise; result is the path itself (persistent root).
        assert result is not None
    finally:
        if test_path.exists():
            test_path.rmdir()


def test_check_not_tmpfs_passes_for_regular_fs(tmp_path: Path) -> None:
    """A path on a regular (non-tmpfs) filesystem must not raise."""
    # tmp_path from pytest is typically on /tmp which may be tmpfs;
    # create a path under the workspace which is guaranteed to be a real FS.
    workspace = Path("/agent/workspace/TensorTorrent")
    if not workspace.exists():
        pytest.skip("workspace not available")
    test_dir = workspace / ".tt_test_check_not_tmpfs"
    test_dir.mkdir(exist_ok=True)
    try:
        _check_not_tmpfs(test_dir)  # Should NOT raise
    finally:
        if test_dir.exists():
            test_dir.rmdir()


# ---------------------------------------------------------------------------
# _resolve_spill_dir precedence: config > TT_SPILL_DIR env > cache fallback
# ---------------------------------------------------------------------------


def test_resolve_spill_dir_config_wins(tmp_path: Path, monkeypatch: Any) -> None:
    """Explicit config_spill_dir takes priority over env and cache_dir."""
    monkeypatch.setenv("TT_ALLOW_TMPFS_SPILL", "1")
    monkeypatch.setenv("TT_SPILL_DIR", str(tmp_path / "env_spill"))
    cache = tmp_path / "cache"
    explicit = tmp_path / "explicit_spill"
    explicit.mkdir(parents=True, exist_ok=True)

    result = _resolve_spill_dir(str(explicit), cache)
    assert str(explicit) in str(result)


def test_resolve_spill_dir_env_wins_over_cache(tmp_path: Path, monkeypatch: Any) -> None:
    """TT_SPILL_DIR env overrides cache fallback when no config is set."""
    monkeypatch.setenv("TT_ALLOW_TMPFS_SPILL", "1")
    env_dir = tmp_path / "env_spill"
    env_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("TT_SPILL_DIR", str(env_dir))
    cache = tmp_path / "cache"

    result = _resolve_spill_dir(None, cache)
    # Result is a per-run sub-directory under env_dir
    assert str(env_dir) in str(result)


def test_resolve_spill_dir_cache_fallback(tmp_path: Path, monkeypatch: Any) -> None:
    """When no config or env, uses <cache_dir>/spill."""
    monkeypatch.setenv("TT_ALLOW_TMPFS_SPILL", "1")
    monkeypatch.delenv("TT_SPILL_DIR", raising=False)
    cache = tmp_path / "cache"

    result = _resolve_spill_dir(None, cache)
    assert str(cache / "spill") in str(result)


def test_resolve_spill_dir_no_config_no_cache_uses_tempdir(tmp_path: Path, monkeypatch: Any) -> None:
    """No config, no env, no cache → legacy tempfile.mkdtemp fallback."""
    monkeypatch.delenv("TT_SPILL_DIR", raising=False)
    monkeypatch.setenv("TT_ALLOW_TMPFS_SPILL", "1")
    result = _resolve_spill_dir(None, None)
    assert Path(result).exists()
    import shutil

    shutil.rmtree(str(result), ignore_errors=True)


# ---------------------------------------------------------------------------
# sweep_orphan_spill_sessions
# ---------------------------------------------------------------------------


def test_sweep_orphan_spill_sessions_removes_dead_pid(tmp_path: Path) -> None:
    """Dead-PID session dir is swept; own-PID dir and unrelated dirs survive."""
    import tensortorrent._native as native

    dead_pid = 4294000000  # guaranteed non-existent PID
    dead_dir = tmp_path / f"tt-spill-{dead_pid}-1"
    dead_dir.mkdir()
    (dead_dir / "spill_data.bin").write_bytes(b"\x00" * 64)

    own_dir = tmp_path / f"tt-spill-{os.getpid()}-1"
    own_dir.mkdir()

    other_dir = tmp_path / "other-unrelated-dir"
    other_dir.mkdir()

    count = native.sweep_orphan_spill_sessions(str(tmp_path))

    assert count == 1, f"expected 1 swept session, got {count}"
    assert not dead_dir.exists(), "dead-PID session dir should have been removed"
    assert own_dir.exists(), "own-PID dir should survive"
    assert other_dir.exists(), "unrelated dir should survive"


def test_sweep_orphan_spill_sessions_empty_dir(tmp_path: Path) -> None:
    """Empty directory → sweep returns 0."""
    import tensortorrent._native as native

    count = native.sweep_orphan_spill_sessions(str(tmp_path))
    assert count == 0
