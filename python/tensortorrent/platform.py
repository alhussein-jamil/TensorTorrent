"""Host OS/arch facts shared by install, library-path setup, and doctor.

Stdlib only. Safe to import before the native extension or PyTorch exist.
Bootstrap and ``tools/native_env.py`` load this file directly so they do not
execute ``tensortorrent.__init__`` (that import requires torch).
"""

from __future__ import annotations

import os
import platform
import sys
from dataclasses import dataclass
from typing import Literal

OsName = Literal["linux", "macos", "windows", "unknown"]
SupportLevel = Literal["production", "development", "unsupported"]

PYTORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"

_ARCH_ALIASES = {
    "x86_64": "x86_64",
    "amd64": "x86_64",
    "aarch64": "aarch64",
    "arm64": "aarch64",
    "armv7l": "armv7",
    "armv8l": "aarch64",
}

_LIBRARY_PATH_VARS: dict[OsName, tuple[str, ...]] = {
    "linux": ("LD_LIBRARY_PATH",),
    "macos": ("DYLD_LIBRARY_PATH", "DYLD_FALLBACK_LIBRARY_PATH"),
    "windows": (),
    "unknown": (),
}


@dataclass(frozen=True)
class HostPlatform:
    """Resolved host identity and TensorTorrent support level."""

    os: OsName
    arch: str
    python: str
    support_level: SupportLevel
    notes: tuple[str, ...]

    @property
    def supported(self) -> bool:
        return self.support_level != "unsupported"

    @property
    def label(self) -> str:
        return f"{self.os}/{self.arch} py{self.python}"


def normalize_arch(machine: str | None = None) -> str:
    raw = (machine or platform.machine() or "unknown").strip().lower()
    return _ARCH_ALIASES.get(raw, raw or "unknown")


def detect_os(sys_platform: str | None = None) -> OsName:
    name = sys_platform if sys_platform is not None else sys.platform
    if name.startswith("linux"):
        return "linux"
    if name == "darwin":
        return "macos"
    if name in {"win32", "cygwin"}:
        return "windows"
    return "unknown"


def detect(
    *,
    sys_platform: str | None = None,
    machine: str | None = None,
    python: str | None = None,
) -> HostPlatform:
    """Classify this host. Keyword overrides keep the logic unit-testable."""

    os_name = detect_os(sys_platform)
    arch = normalize_arch(machine)
    py = python if python is not None else platform.python_version()
    if os_name == "linux":
        return HostPlatform(os="linux", arch=arch, python=py, support_level="production", notes=())
    if os_name == "macos":
        return HostPlatform(
            os="macos",
            arch=arch,
            python=py,
            support_level="development",
            notes=(
                "CPU and source builds are supported.",
                "Apple GPU (MPS) is not a TensorTorrent backend.",
                "process_workers requires Linux.",
            ),
        )
    if os_name == "windows":
        return HostPlatform(
            os="windows",
            arch=arch,
            python=py,
            support_level="unsupported",
            notes=("Windows is not supported. Use Linux or macOS.",),
        )
    return HostPlatform(
        os=os_name,
        arch=arch,
        python=py,
        support_level="unsupported",
        notes=(f"platform {os_name!r} is not supported.",),
    )


def supports_process_workers(*, sys_platform: str | None = None) -> bool:
    """Linux fork semantics only. macOS/Windows must keep ``process_workers=0``."""

    return detect_os(sys_platform) == "linux"


def native_library_path_vars(*, sys_platform: str | None = None) -> tuple[str, ...]:
    return _LIBRARY_PATH_VARS[detect_os(sys_platform)]


def apply_native_library_path(libdir: str, *, sys_platform: str | None = None) -> None:
    """Prepend ``libdir`` on the loader path variables this host honors."""

    if not libdir:
        return
    for key in native_library_path_vars(sys_platform=sys_platform):
        os.environ[key] = _prepend_path(libdir, os.environ.get(key))


def emit_native_library_exports(libdir: str, *, sys_platform: str | None = None) -> str:
    """POSIX ``export`` lines for Make/CI. Empty ``libdir`` emits nothing."""

    if not libdir:
        return ""
    lines: list[str] = []
    for key in native_library_path_vars(sys_platform=sys_platform):
        value = _prepend_path(libdir, os.environ.get(key))
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        lines.append(f'export {key}="{escaped}"')
    return "\n".join(lines)


def _prepend_path(libdir: str, prev: str | None) -> str:
    if not prev:
        return libdir
    parts = prev.split(os.pathsep)
    if parts and parts[0] == libdir:
        return prev
    return f"{libdir}{os.pathsep}{prev}"


def torch_index_url(*, flavor: str = "cpu") -> str | None:
    """Wheel extra-index for a stock CPU build. Accelerator wheels: install torch yourself."""

    if flavor == "cpu":
        return PYTORCH_CPU_INDEX
    return None
