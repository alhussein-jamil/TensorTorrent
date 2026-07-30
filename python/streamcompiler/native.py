"""Native extension loader.

The public runtime path requires the Rust extension built via maturin.
"""

from __future__ import annotations

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
    raise ImportError(
        "streamcompiler native extension is unavailable. "
        "Build it with `uv run maturin develop` or `uv sync` "
        f"(import error: {_NATIVE_ERROR})"
    )
