"""Runtime compatibility checks for supported Python / PyTorch ranges."""

from __future__ import annotations

from types import ModuleType

TORCH_MIN = (2, 4)


def parse_torch_version(version: str) -> tuple[int, int]:
    """Return ``(major, minor)`` from a torch version string (ignores patch/local)."""
    core = version.split("+", 1)[0].split("-", 1)[0].strip()
    if not core:
        raise ValueError(f"empty torch version: {version!r}")
    parts = core.split(".")
    try:
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
    except ValueError as exc:
        raise ValueError(f"unparsable torch version: {version!r}") from exc
    return major, minor


def torch_meets_minimum(version: str, *, minimum: tuple[int, int] = TORCH_MIN) -> bool:
    return parse_torch_version(version) >= minimum


def require_torch() -> ModuleType:
    """Import torch and ensure it meets the supported floor (``>= 2.4``).

    Install your preferred torch build first (CPU / CUDA / ROCm from pytorch.org);
    ``pip install tensortorrent`` will reuse it when it already satisfies ``torch>=2.4``.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "tensortorrent requires PyTorch >= 2.4. Install the torch build you want "
            "first (see https://pytorch.org/get-started/locally/), then "
            "`pip install tensortorrent`."
        ) from exc

    version = str(getattr(torch, "__version__", ""))
    if not torch_meets_minimum(version):
        raise ImportError(
            f"tensortorrent requires PyTorch >= {TORCH_MIN[0]}.{TORCH_MIN[1]} "
            f"(found {version}). Upgrade torch or install a supported build from "
            "https://pytorch.org/get-started/locally/."
        )
    return torch


def torch_compat_line() -> str:
    """One-line status for ``tensortorrent doctor`` (calls :func:`require_torch`)."""
    torch = require_torch()
    version = str(torch.__version__)
    return f"torch: {version} (ok, requires >={TORCH_MIN[0]}.{TORCH_MIN[1]})"
