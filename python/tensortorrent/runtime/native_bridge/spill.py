"""Native spill directory resolution and setup."""

from __future__ import annotations

import contextlib
import os
import tempfile as _stdlib_tempfile
from pathlib import Path
from types import ModuleType
from typing import Any

from tensortorrent.errors import ConfigurationError

_ORPHAN_SWEEP_DONE = False


def _spill_tempfile() -> ModuleType:
    """Return the ``tempfile`` module used for spill dirs.

    Resolved via the package so tests can monkeypatch
    ``tensortorrent.runtime.native_bridge.tempfile.mkdtemp``.
    """
    import tensortorrent.runtime.native_bridge as nb

    mod = getattr(nb, "tempfile", None)
    return mod if isinstance(mod, ModuleType) else _stdlib_tempfile


def _resolve_spill_root(config_spill_dir: str | None, cache_dir: Path | None) -> Path | None:
    """Resolve the persistent spill root directory.

    Returns the root directory when a persistent root is configured, or None
    to signal that the caller should use a plain ``tempfile.mkdtemp()`` call
    (legacy fallback; tests monkeypatch ``tempfile.mkdtemp`` on this package).

    Raises :class:`~tensortorrent.errors.ConfigurationError` when the resolved
    directory lives on a tmpfs filesystem, unless ``TT_ALLOW_TMPFS_SPILL=1`` is set.
    """
    # 1. Explicit config value
    if config_spill_dir:
        chosen = Path(config_spill_dir)
    # 2. Environment variable
    elif os.environ.get("TT_SPILL_DIR"):
        chosen = Path(os.environ["TT_SPILL_DIR"])
    # 3. Cache dir sub-path
    elif cache_dir is not None:
        chosen = cache_dir / "spill"
    else:
        # 4. No persistent root — caller uses legacy tempfile.mkdtemp()
        return None

    chosen.mkdir(parents=True, exist_ok=True)

    # tmpfs refusal via /proc/mounts longest-prefix fstype check
    if os.environ.get("TT_ALLOW_TMPFS_SPILL", "0") != "1":
        _check_not_tmpfs(chosen)

    return chosen


def _resolve_spill_dir(config_spill_dir: str | None, cache_dir: Path | None) -> Path:
    """Resolve the per-run spill directory (legacy-compatible entry point).

    This is the public monkeypatch point used by tests. It calls
    ``_resolve_spill_root`` and, when no persistent root is found, falls back to
    ``tempfile.mkdtemp()`` so monkeypatching ``tempfile.mkdtemp`` on this module
    correctly intercepts the legacy fallback path.
    """
    root = _resolve_spill_root(config_spill_dir, cache_dir)
    tempfile = _spill_tempfile()
    if root is None:
        return Path(tempfile.mkdtemp(prefix="tt_native_spill_"))
    # Create a per-run sub-session under the persistent root.
    return Path(tempfile.mkdtemp(prefix="tt_native_spill_", dir=root))


def _check_not_tmpfs(path: Path) -> None:
    """Raise ConfigurationError if *path* is on a tmpfs/ramfs mount."""
    try:
        mounts_text = Path("/proc/mounts").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return  # Cannot read mounts — skip check

    resolved = str(path.resolve())
    best_prefix = ""
    best_fstype = ""
    for line in mounts_text.splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        mountpoint = parts[1]
        fstype = parts[2]
        if resolved.startswith(mountpoint) and len(mountpoint) > len(best_prefix):
            best_prefix = mountpoint
            best_fstype = fstype

    if best_fstype in ("tmpfs", "ramfs"):
        raise ConfigurationError(
            f"Spill directory {resolved!r} is on a {best_fstype!r} filesystem "
            "(data will not survive a reboot and may exhaust RAM). "
            "Set TT_ALLOW_TMPFS_SPILL=1 to override, or choose a persistent path."
        )


def _setup_native_spill(native: Any, native_ctx: Any, executor: Any) -> Path:
    """Configure the native context with the resolved spill directory and budget.

    Called once per forward pass when spill callbacks are needed.
    Returns the ephemeral per-run spill directory that must be cleaned up after run.
    """
    global _ORPHAN_SWEEP_DONE  # noqa: PLW0603

    config_spill_dir: str | None = getattr(executor, "_config_spill_dir", None)
    cache_dir: Path | None = getattr(executor, "_config_cache_dir", None)

    # Resolve the persistent root (may be None → legacy path).
    spill_root = _resolve_spill_root(config_spill_dir, cache_dir)

    # Sweep orphans once per process when a persistent root is known.
    if spill_root is not None and not _ORPHAN_SWEEP_DONE:
        _ORPHAN_SWEEP_DONE = True
        if hasattr(native, "sweep_orphan_spill_sessions"):
            with contextlib.suppress(Exception):
                native.sweep_orphan_spill_sessions(str(spill_root))

    # Create the per-run ephemeral directory. When no persistent root was
    # configured, fall back to tempfile.mkdtemp() so package-level monkeypatches
    # of tempfile.mkdtemp still intercept the legacy path.
    tempfile = _spill_tempfile()
    if spill_root is not None:
        run_spill_dir = Path(tempfile.mkdtemp(prefix="tt_native_spill_", dir=spill_root))
    else:
        run_spill_dir = Path(tempfile.mkdtemp(prefix="tt_native_spill_"))

    native_ctx.set_spill_dir(str(run_spill_dir))

    # Budget: prefer explicit config, else resolve from the dir we'll use.
    max_spill: int | None = getattr(executor, "_config_max_total_spill_bytes", None)
    if max_spill is None:
        from tensortorrent.hardware import budget as _budget

        budget_path = spill_root if spill_root is not None else run_spill_dir
        budget_result = _budget.resolve_disk_budget(budget_path)
        max_spill = budget_result.allowed_bytes
    if hasattr(native_ctx, "set_spill_budget_bytes"):
        with contextlib.suppress(Exception):
            native_ctx.set_spill_budget_bytes(max_spill)

    stall_s: float = getattr(executor, "_config_stall_timeout_s", 300.0)
    if hasattr(native_ctx, "set_stall_timeout_secs"):
        with contextlib.suppress(Exception):
            native_ctx.set_stall_timeout_secs(stall_s)

    return run_spill_dir


def _merge_native_streaming_io_intervals(executor: Any) -> None:
    store = getattr(executor, "parameter_store", None)
    native = getattr(store, "_native_store", None)
    if store is None or native is None or not hasattr(native, "io_intervals"):
        return
    origin = float(getattr(store, "_native_io_origin", 0.0) or 0.0)
    if origin <= 0.0:
        return
    from tensortorrent.runtime.tensor_store import IoInterval

    intervals = list(getattr(store, "_io_intervals", []))
    for start, end, nbytes in native.io_intervals():
        intervals.append(
            IoInterval(
                name="native_prefetch",
                start_s=origin + float(start),
                end_s=origin + float(end),
                nbytes=int(nbytes),
            )
        )
    store._io_intervals = intervals
