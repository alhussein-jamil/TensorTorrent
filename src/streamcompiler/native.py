"""Native extension loader.

The public runtime path requires the Rust extension built via maturin.
Set ``STREAMCOMPILER_ALLOW_PYTHON_RUNTIME=1`` only for migration benchmarks.
"""

from __future__ import annotations

import os
from types import ModuleType

_NATIVE: ModuleType | None = None
_NATIVE_ERROR: str | None = None

try:
    from streamcompiler import _native as _imported_native  # type: ignore[attr-defined]

    _NATIVE = _imported_native
except Exception as exc:  # pragma: no cover - import failure path
    _NATIVE_ERROR = str(exc)


def native_available() -> bool:
    return _NATIVE is not None and bool(getattr(_NATIVE, "native_available", lambda: False)())


def require_native() -> ModuleType:
    """Return the native module or raise a clear ImportError."""
    if _NATIVE is not None:
        return _NATIVE
    allow = os.environ.get("STREAMCOMPILER_ALLOW_PYTHON_RUNTIME", "").strip() in {"1", "true", "yes"}
    if allow:
        raise ImportError(
            "native extension unavailable but STREAMCOMPILER_ALLOW_PYTHON_RUNTIME is set; "
            "callers must use the explicit Python fallback path"
        )
    raise ImportError(
        "streamcompiler native extension is unavailable. "
        "Build it with `maturin develop` or `pip install .` "
        f"(import error: {_NATIVE_ERROR})"
    )


def allow_python_runtime() -> bool:
    return os.environ.get("STREAMCOMPILER_ALLOW_PYTHON_RUNTIME", "").strip() in {"1", "true", "yes"}
