from streamcompiler.backends.base import ExecutionBackend
from streamcompiler.backends.cpu import CpuBackend
from streamcompiler.backends.cuda import CudaBackend
from streamcompiler.backends.mock_accel import MockAccelBackend
from streamcompiler.backends.rocm import RocmBackend

# Production discovery order. MPS/SYCL/OpenCL/Vulkan stubs removed — unsupported claims.
# CUDA/ROCm remain for discovery; execution readiness is hardware-gated (see docs/PRODUCT.md).
_BACKEND_CTORS: tuple[type[ExecutionBackend], ...] = (
    CpuBackend,
    CudaBackend,
    RocmBackend,
    MockAccelBackend,
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


def backend_id_for_resource(resource_id: str) -> str:
    """Map a schedule resource name to the owning backend id.

    Order matters: ROCm / mock must win before generic ``cuda`` / ``gpu``
    substrings so ``rocm_gpu_0`` is not mis-routed.
    """
    name = resource_id.lower()
    if "mock_accel" in name or name.startswith("mock_"):
        return "mock_accel"
    if "rocm" in name:
        return "rocm"
    if "cuda" in name:
        return "cuda"
    if name.startswith("gpu"):
        return "cuda"
    return "cpu"


__all__ = [
    "CpuBackend",
    "CudaBackend",
    "ExecutionBackend",
    "MockAccelBackend",
    "RocmBackend",
    "all_backends",
    "available_backends",
    "backend_by_id",
    "backend_id_for_resource",
]
