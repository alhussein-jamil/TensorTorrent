"""Resource budget resolver — single source of truth for memory and CPU limits.

Precedence chain for host memory:
  1. explicit (caller-supplied value wins unconditionally)
  2. cgroup v2 available (memory.max or memory.high minus memory.current)
  3. cgroup v1 available (memory.limit_in_bytes minus memory.usage_in_bytes)
  4. OS available (psutil.virtual_memory().available)
  5. OS total (psutil.virtual_memory().total) — last-resort fallback, noted

For CPU count:
  1. explicit
  2. min(sched_getaffinity length, cgroup v2 cpu.max quota, cgroup v1 cfs quota, os.cpu_count())

For disk (spill path):
  1. explicit
  2. 80% of shutil.disk_usage().free

For device VRAM:
  1. explicit
  2. live free − headroom  (source = os_available)
  3. total*0.9 − headroom  (source = total_fallback, noted)

All resolver functions accept optional root paths for cgroup/proc trees so they
are fully testable without touching the live filesystem.
"""

from __future__ import annotations

import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

_KindStr = Literal["explicit", "cgroup_v2", "cgroup_v1", "os_available", "total_fallback"]

_MiB = 1 << 20
_GiB = 1 << 30

_256_MiB = 256 * _MiB
_128_MiB = 128 * _MiB
_2_GiB = 2 * _GiB
_768_MiB = 768 * _MiB


@dataclass(frozen=True)
class BudgetSource:
    """Provenance tag attached to every resolved budget value.

    kind is one of: "explicit", "cgroup_v2", "cgroup_v1",
                    "os_available", "total_fallback".
    detail is a human-readable description of the underlying query.
    """

    kind: _KindStr
    detail: str


@dataclass(frozen=True)
class ResolvedBudget:
    """Result of a budget resolution call.

    total_bytes   — physical or cgroup ceiling (for display)
    allowed_bytes — what workloads should actually use
    reserved_bytes — bytes withheld from allowed
    source        — where the numbers came from
    notes         — advisory strings (warnings, caveats)
    """

    total_bytes: int
    allowed_bytes: int
    reserved_bytes: int
    source: BudgetSource
    notes: tuple[str, ...]


# ---------------------------------------------------------------------------
# Internal cgroup helpers
# ---------------------------------------------------------------------------


def _read_cgroup_v2_memory(cgroup_root: str = "/sys/fs/cgroup") -> tuple[int | None, str]:
    """Return (available_bytes, detail) from cgroup v2, or (None, reason)."""
    root = Path(cgroup_root)
    max_path = root / "memory.max"
    high_path = root / "memory.high"
    current_path = root / "memory.current"

    if not max_path.exists() and not high_path.exists():
        return None, "cgroup_v2 not present"

    def _read_int(p: Path) -> int | None:
        try:
            text = p.read_text(encoding="utf-8").strip()
            if text == "max":
                return None  # unlimited
            return int(text)
        except (OSError, ValueError):
            return None

    current = _read_int(current_path)
    if current is None:
        current = 0

    limits: list[int] = []
    for p in (max_path, high_path):
        v = _read_int(p)
        if v is not None:
            limits.append(v)

    if not limits:
        return None, "cgroup_v2 limits both 'max' (unlimited)"

    ceiling = min(limits)
    available = max(0, ceiling - current)
    detail = f"cgroup_v2 ceiling={ceiling} current={current}"
    return available, detail


def _read_cgroup_v1_memory(cgroup_root: str = "/sys/fs/cgroup") -> tuple[int | None, str]:
    """Return (available_bytes, detail) from cgroup v1, or (None, reason)."""
    root = Path(cgroup_root) / "memory"
    limit_path = root / "memory.limit_in_bytes"
    usage_path = root / "memory.usage_in_bytes"

    if not limit_path.exists():
        return None, "cgroup_v1 not present"

    def _read_int(p: Path) -> int | None:
        try:
            return int(p.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            return None

    limit = _read_int(limit_path)
    if limit is None:
        return None, "cgroup_v1 limit unreadable"

    # Values >= 0x7FFFFFFFFFFFF000 mean unlimited in cgroup v1.
    if limit >= 0x7FFFFFFFFFFFF000:
        return None, f"cgroup_v1 limit={limit:#x} (unlimited sentinel)"

    usage = _read_int(usage_path) or 0
    available = max(0, limit - usage)
    detail = f"cgroup_v1 limit={limit} usage={usage}"
    return available, detail


def _cgroup_cpu_quota_v2(cgroup_root: str = "/sys/fs/cgroup") -> int | None:
    """Return effective CPU count from cgroup v2 cpu.max, or None if unlimited."""
    cpu_max = Path(cgroup_root) / "cpu.max"
    if not cpu_max.exists():
        return None
    try:
        text = cpu_max.read_text(encoding="utf-8").strip()
        parts = text.split()
        if len(parts) < 2 or parts[0] == "max":
            return None
        quota = int(parts[0])
        period = int(parts[1])
        if period <= 0:
            return None
        return math.ceil(quota / period)
    except (OSError, ValueError):
        return None


def _cgroup_cpu_quota_v1(cgroup_root: str = "/sys/fs/cgroup") -> int | None:
    """Return effective CPU count from cgroup v1 CFS quota, or None if unlimited."""
    quota_path = Path(cgroup_root) / "cpu" / "cpu.cfs_quota_us"
    period_path = Path(cgroup_root) / "cpu" / "cpu.cfs_period_us"

    def _read_int(p: Path) -> int | None:
        try:
            return int(p.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            return None

    quota = _read_int(quota_path)
    if quota is None or quota == -1:
        return None  # unlimited
    period = _read_int(period_path)
    if period is None or period <= 0:
        return None
    return math.ceil(quota / period)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def is_wsl2(proc_version_path: str = "/proc/version") -> bool:
    """Return True when running inside WSL2 (Windows Subsystem for Linux)."""
    try:
        text = Path(proc_version_path).read_text(encoding="utf-8", errors="replace")
        return "microsoft" in text.lower()
    except OSError:
        return False


def default_vram_headroom_bytes(display_active: bool) -> int:
    """How many bytes to reserve for the display/OS on a GPU.

    768 MiB when a display is attached (conservative), 256 MiB otherwise.
    TT_VRAM_HEADROOM_BYTES env var overrides both.
    """
    env = os.environ.get("TT_VRAM_HEADROOM_BYTES")
    if env:
        try:
            return max(0, int(env))
        except ValueError:
            pass
    return _768_MiB if display_active else _256_MiB


def vram_capacity_floor_bytes(total_bytes: int, headroom_bytes: int) -> int:
    """Physical-capacity floor for a VRAM planning budget.

    ``total - headroom`` (clamped to 0). Used to shield the planner from
    transient live-free readings when a framework's caching allocator
    (e.g. PyTorch) is holding VRAM that will be reclaimed before execution.
    Disabled by setting ``TT_DISABLE_VRAM_CAPACITY_FLOOR=1`` in the env.
    """
    if os.environ.get("TT_DISABLE_VRAM_CAPACITY_FLOOR", "").lower() in {"1", "true", "yes"}:
        return 0
    return max(0, int(total_bytes) - max(0, int(headroom_bytes)))


def _default_reserve(raw: int) -> int:
    """5 % of raw, clamped to [256 MiB, 2 GiB], env-overridable."""
    env = os.environ.get("TT_HOST_MEMORY_RESERVE_BYTES")
    if env:
        try:
            return max(0, int(env))
        except ValueError:
            pass
    return int(max(_256_MiB, min(_2_GiB, raw * 0.05)))


def resolve_host_memory_budget(
    explicit: int | None = None,
    *,
    reserve_bytes: int | None = None,
    cgroup_root: str = "/sys/fs/cgroup",
    proc_version_path: str = "/proc/version",
) -> ResolvedBudget:
    """Resolve the host memory budget with full provenance.

    Precedence:
      explicit > min(cgroup_v2, cgroup_v1, os_available) > os_total
    """
    import psutil

    vm = psutil.virtual_memory()

    if explicit is not None:
        raw = explicit
        source = BudgetSource(kind="explicit", detail=f"caller-supplied {explicit}")
        reserve = reserve_bytes if reserve_bytes is not None else _default_reserve(raw)
        allowed = max(_128_MiB, raw - reserve)
        return ResolvedBudget(
            total_bytes=vm.total,
            allowed_bytes=allowed,
            reserved_bytes=reserve,
            source=source,
            notes=(),
        )

    candidates: list[tuple[int, BudgetSource]] = []
    notes: list[str] = []

    v2_avail, v2_detail = _read_cgroup_v2_memory(cgroup_root)
    if v2_avail is not None:
        candidates.append((v2_avail, BudgetSource(kind="cgroup_v2", detail=v2_detail)))

    v1_avail, v1_detail = _read_cgroup_v1_memory(cgroup_root)
    if v1_avail is not None:
        candidates.append((v1_avail, BudgetSource(kind="cgroup_v1", detail=v1_detail)))

    os_avail = int(vm.available)
    candidates.append(
        (os_avail, BudgetSource(kind="os_available", detail=f"psutil.virtual_memory().available={os_avail}"))
    )

    if candidates:
        raw, source = min(candidates, key=lambda t: t[0])
    else:
        raw = int(vm.total)
        source = BudgetSource(
            kind="total_fallback",
            detail=f"psutil.virtual_memory().total={vm.total}; no available metric",
        )
        notes.append("budget derived from total RAM; available query unavailable")

    reserve = reserve_bytes if reserve_bytes is not None else _default_reserve(raw)
    allowed = max(_128_MiB, raw - reserve)

    return ResolvedBudget(
        total_bytes=int(vm.total),
        allowed_bytes=allowed,
        reserved_bytes=reserve,
        source=source,
        notes=tuple(notes),
    )


def resolve_cpu_budget(
    explicit: int | None = None,
    *,
    cgroup_root: str = "/sys/fs/cgroup",
) -> tuple[int, BudgetSource]:
    """Resolve effective CPU count with provenance.

    Precedence: explicit > min(affinity, cgroup_v2_quota, cgroup_v1_quota, os.cpu_count())
    """
    if explicit is not None:
        return explicit, BudgetSource(kind="explicit", detail=f"caller-supplied {explicit}")

    candidates: list[tuple[int, str]] = []

    if hasattr(os, "sched_getaffinity"):
        try:
            affinity_count = len(os.sched_getaffinity(0))
            candidates.append((affinity_count, f"sched_getaffinity(0)={affinity_count}"))
        except OSError:
            pass

    v2_quota = _cgroup_cpu_quota_v2(cgroup_root)
    if v2_quota is not None:
        candidates.append((v2_quota, f"cgroup_v2_cpu_max_quota={v2_quota}"))

    v1_quota = _cgroup_cpu_quota_v1(cgroup_root)
    if v1_quota is not None:
        candidates.append((v1_quota, f"cgroup_v1_cfs_quota={v1_quota}"))

    os_count = os.cpu_count() or 1
    candidates.append((os_count, f"os.cpu_count()={os_count}"))

    count, detail = min(candidates, key=lambda t: t[0])

    # Determine source kind
    if "cgroup_v2" in detail:
        kind: _KindStr = "cgroup_v2"
    elif "cgroup_v1" in detail:
        kind = "cgroup_v1"
    elif "affinity" in detail or "cpu_count" in detail.lower() or "os.cpu_count" in detail:
        kind = "os_available"
    else:
        kind = "os_available"

    return count, BudgetSource(kind=kind, detail=detail)


def resolve_disk_budget(
    path: str | os.PathLike[str],
    explicit: int | None = None,
) -> ResolvedBudget:
    """Resolve the disk budget for a given path.

    allowed = explicit if provided, else 80% of free space.
    """
    try:
        usage = shutil.disk_usage(str(path))
        free = int(usage.free)
        total = int(usage.total)
    except OSError:
        free = 0
        total = 0

    if explicit is not None:
        allowed = explicit
        source = BudgetSource(kind="explicit", detail=f"caller-supplied {explicit}")
        reserve = max(0, free - allowed) if free > 0 else 0
    else:
        allowed = int(free * 0.8)
        reserve = free - allowed
        source = BudgetSource(
            kind="os_available",
            detail=f"shutil.disk_usage({path!r}).free={free} → allowed=80%",
        )

    return ResolvedBudget(
        total_bytes=total,
        allowed_bytes=max(0, allowed),
        reserved_bytes=max(0, reserve),
        source=source,
        notes=(),
    )


def resolve_device_memory_budget(
    total_bytes: int,
    free_bytes: int | None,
    explicit: int | None,
    headroom_bytes: int,
) -> ResolvedBudget:
    """Resolve device (VRAM) memory budget.

    Precedence:
      1. explicit  (source = explicit)
      2. free_bytes - headroom  (source = os_available)
      3. total_bytes*0.9 - headroom  (source = total_fallback, noted)

    Result is never below 0.
    """
    if explicit is not None:
        allowed = max(0, explicit)
        reserve = max(0, total_bytes - allowed)
        return ResolvedBudget(
            total_bytes=total_bytes,
            allowed_bytes=allowed,
            reserved_bytes=reserve,
            source=BudgetSource(kind="explicit", detail=f"caller-supplied {explicit}"),
            notes=(),
        )

    if free_bytes is not None:
        allowed = max(0, free_bytes - headroom_bytes)
        reserve = headroom_bytes
        return ResolvedBudget(
            total_bytes=total_bytes,
            allowed_bytes=allowed,
            reserved_bytes=reserve,
            source=BudgetSource(
                kind="os_available",
                detail=f"live_free={free_bytes} − headroom={headroom_bytes}",
            ),
            notes=(),
        )

    # Fallback: use 90% of total minus headroom
    base = int(total_bytes * 0.9)
    allowed = max(0, base - headroom_bytes)
    reserve = total_bytes - allowed
    return ResolvedBudget(
        total_bytes=total_bytes,
        allowed_bytes=allowed,
        reserved_bytes=reserve,
        source=BudgetSource(
            kind="total_fallback",
            detail=f"total*0.9={base} − headroom={headroom_bytes}; live free unavailable",
        ),
        notes=("live free memory was unavailable; budget derived from total",),
    )
