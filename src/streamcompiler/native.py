"""Native extension loader.

The public runtime path requires the Rust extension built via maturin.
Set ``STREAMCOMPILER_DEV_PYTHON_RUNTIME=1`` only for developer oracle runs.
``STREAMCOMPILER_ALLOW_PYTHON_RUNTIME`` remains accepted as a deprecated alias.
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
    if allow_python_runtime():
        raise ImportError(
            "native extension unavailable but STREAMCOMPILER_DEV_PYTHON_RUNTIME is set; "
            "callers must use the explicit Python fallback path"
        )
    raise ImportError(
        "streamcompiler native extension is unavailable. "
        "Build it with `maturin develop` or `pip install .` "
        f"(import error: {_NATIVE_ERROR})"
    )


def allow_python_runtime() -> bool:
    """Developer-only Python DAG fallback. Never activates silently."""
    for key in ("STREAMCOMPILER_DEV_PYTHON_RUNTIME", "STREAMCOMPILER_ALLOW_PYTHON_RUNTIME"):
        if os.environ.get(key, "").strip() in {"1", "true", "yes"}:
            return True
    return False
