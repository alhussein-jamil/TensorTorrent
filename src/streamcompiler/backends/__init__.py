from streamcompiler.backends.base import ExecutionBackend
from streamcompiler.backends.cpu import CpuBackend
from streamcompiler.backends.cuda import CudaBackend
from streamcompiler.backends.mps import MpsBackend
from streamcompiler.backends.opencl_vulkan import OpenCLBackend, VulkanBackend
from streamcompiler.backends.rocm import RocmBackend
from streamcompiler.backends.sycl import SyclBackend

# Order is registration order for discovery; planner decisions use measured costs.
_BACKEND_CTORS: tuple[type[ExecutionBackend], ...] = (
    CpuBackend,
    CudaBackend,
    RocmBackend,
    MpsBackend,
    SyclBackend,
    OpenCLBackend,
    VulkanBackend,
)


def all_backends() -> list[ExecutionBackend]:
    return [ctor() for ctor in _BACKEND_CTORS]


def available_backends() -> list[ExecutionBackend]:
    return [b for b in all_backends() if b.available()]


def backend_by_id(backend_id: str) -> ExecutionBackend | None:
    for backend in all_backends():
        if backend.backend_id == backend_id:
            return backend
    return None


__all__ = [
    "CpuBackend",
    "CudaBackend",
    "ExecutionBackend",
    "MpsBackend",
    "OpenCLBackend",
    "RocmBackend",
    "SyclBackend",
    "VulkanBackend",
    "all_backends",
    "available_backends",
    "backend_by_id",
]
