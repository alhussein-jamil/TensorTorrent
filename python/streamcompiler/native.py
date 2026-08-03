"""Native extension loader.

The public runtime path requires the Rust extension built via maturin. Loading is
performed through ``importlib`` so a missing extension reports the actual module
error rather than a misleading partially-initialized package/circular-import
message.
"""

from __future__ import annotations

import importlib
from types import ModuleType

_NATIVE: ModuleType | None = None
_NATIVE_ERROR: str | None = None

try:
    _NATIVE = importlib.import_module("streamcompiler._native")
except Exception as exc:  # pragma: no cover - import failure path
    _NATIVE_ERROR = f"{type(exc).__name__}: {exc}"


def native_available() -> bool:
    return _NATIVE is not None and bool(getattr(_NATIVE, "native_available", lambda: False)())


def require_native() -> ModuleType:
    """Return the native module or raise a clear ImportError."""
    if _NATIVE is not None:
        return _NATIVE
    raise ImportError(
        "streamcompiler native extension is unavailable. "
        "Build it with `uv run maturin develop --release` or install a native wheel "
        f"(import error: {_NATIVE_ERROR})"
    )
