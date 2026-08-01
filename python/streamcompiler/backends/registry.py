"""Backend registry with safe third-party plugin discovery."""

from __future__ import annotations

import os
from importlib import metadata
from threading import RLock
from typing import Any

from streamcompiler.backends.base import ExecutionBackend

ENTRY_POINT_GROUP = "streamcompiler.backends"

_LOCK = RLock()
_PLUGIN_FACTORIES: tuple[Any, ...] | None = None
_PLUGIN_ERRORS: dict[str, str] = {}


def _entry_points() -> list[Any]:
    discovered = metadata.entry_points()
    if hasattr(discovered, "select"):
        return list(discovered.select(group=ENTRY_POINT_GROUP))
    getter = getattr(discovered, "get", None)
    if callable(getter):
        return list(getter(ENTRY_POINT_GROUP, ()))
    return []


def _coerce_backend(loaded: Any, *, name: str) -> ExecutionBackend:
    candidate = loaded
    if isinstance(candidate, type) or not isinstance(candidate, ExecutionBackend) and callable(candidate):
        candidate = candidate()
    if not isinstance(candidate, ExecutionBackend):
        raise TypeError(
            f"entry point {name!r} must load an ExecutionBackend, subclass, or zero-argument factory; "
            f"got {type(candidate).__name__}"
        )
    if not isinstance(candidate.backend_id, str) or not candidate.backend_id.strip():
        raise TypeError(f"entry point {name!r} returned a backend with an invalid backend_id")
    return candidate


def plugin_backends(*, refresh: bool = False) -> list[ExecutionBackend]:
    """Load external backends without letting one broken plugin break discovery.

    Plugins register an entry point in the ``streamcompiler.backends`` group.
    Set ``STREAMCOMPILER_DISABLE_BACKEND_PLUGINS=1`` for hermetic deployments.
    """
    global _PLUGIN_FACTORIES
    if os.environ.get("STREAMCOMPILER_DISABLE_BACKEND_PLUGINS", "").lower() in {"1", "true", "yes"}:
        return []
    with _LOCK:
        if refresh:
            _PLUGIN_FACTORIES = None
            _PLUGIN_ERRORS.clear()
        if _PLUGIN_FACTORIES is None:
            factories: list[Any] = []
            seen_names: set[str] = set()
            for entry_point in sorted(_entry_points(), key=lambda ep: (ep.name, ep.value)):
                label = f"{entry_point.name}={entry_point.value}"
                if label in seen_names:
                    continue
                seen_names.add(label)
                try:
                    loaded = entry_point.load()
                    # Validate once, retain the factory/object for fresh instances.
                    _coerce_backend(loaded, name=label)
                    factories.append((label, loaded))
                except Exception as exc:  # noqa: BLE001 - plugin isolation boundary
                    _PLUGIN_ERRORS[label] = f"{type(exc).__name__}: {exc}"
            _PLUGIN_FACTORIES = tuple(factories)

        result: list[ExecutionBackend] = []
        for label, loaded in _PLUGIN_FACTORIES:
            try:
                result.append(_coerce_backend(loaded, name=label))
            except Exception as exc:  # noqa: BLE001
                _PLUGIN_ERRORS[label] = f"{type(exc).__name__}: {exc}"
        return result


def plugin_errors() -> dict[str, str]:
    with _LOCK:
        return dict(_PLUGIN_ERRORS)
