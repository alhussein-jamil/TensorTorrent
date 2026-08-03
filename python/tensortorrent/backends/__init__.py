from __future__ import annotations

from tensortorrent.backends.base import ExecutionBackend
from tensortorrent.backends.cpu import CpuBackend
from tensortorrent.backends.cuda import CudaBackend
from tensortorrent.backends.mock_accel import MockAccelBackend
from tensortorrent.backends.registry import plugin_backends, plugin_errors
from tensortorrent.backends.rocm import RocmBackend
from tensortorrent.backends.xpu import XpuBackend

# Production discovery order. Every accelerator is capability-gated; absent
# runtimes never create fake devices. Third-party backends are appended through
# the ``tensortorrent.backends`` entry-point group.
_BUILTIN_CTORS: tuple[type[ExecutionBackend], ...] = (
    CpuBackend,
    CudaBackend,
    RocmBackend,
    XpuBackend,
    MockAccelBackend,
)


def all_backends(*, include_plugins: bool = True) -> list[ExecutionBackend]:
    backends = [ctor() for ctor in _BUILTIN_CTORS]
    if include_plugins:
        backends.extend(plugin_backends())
    # Built-ins win duplicate ids; duplicate plugins are isolated and omitted.
    unique: dict[str, ExecutionBackend] = {}
    for backend in backends:
        unique.setdefault(backend.backend_id, backend)
    return list(unique.values())


def available_backends() -> list[ExecutionBackend]:
    import logging

    log = logging.getLogger("tensortorrent.backends")
    available: list[ExecutionBackend] = []
    for backend in all_backends():
        try:
            if backend.available():
                available.append(backend)
        except Exception as exc:  # noqa: BLE001 - one optional backend cannot poison discovery
            log.warning("backend %s availability probe failed: %s", backend.backend_id, exc)
            continue
    return available


def backend_by_id(backend_id: str) -> ExecutionBackend | None:
    for backend in all_backends():
        if backend.backend_id == backend_id:
            return backend
    return None


def backend_id_for_resource(resource_id: str) -> str:
    """Map standard resource names to their owning backend id.

    Custom plugins should use resource names prefixed by their ``backend_id`` or
    attach the backend explicitly through the resource graph. Unknown names fall
    back to CPU only for host resources.
    """
    name = resource_id.lower()
    if "mock_accel" in name or name.startswith("mock_"):
        return "mock_accel"
    if "rocm" in name:
        return "rocm"
    if "xpu" in name or name.startswith("intel_gpu"):
        return "xpu"
    if "cuda" in name:
        return "cuda"
    if name.startswith("gpu"):
        return "cuda"
    for backend in all_backends():
        if name.startswith(f"{backend.backend_id.lower()}_"):
            return backend.backend_id
    return "cpu"


__all__ = [
    "CpuBackend",
    "CudaBackend",
    "ExecutionBackend",
    "MockAccelBackend",
    "RocmBackend",
    "XpuBackend",
    "all_backends",
    "available_backends",
    "backend_by_id",
    "backend_id_for_resource",
    "plugin_errors",
]
