"""OpenCL / Vulkan discovery stubs.

These backends are registered for capability probing on production machines.
They report unavailable until a concrete runtime is present and validated.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from streamcompiler.backends.base import (
    BenchmarkResult,
    CompiledRegion,
    ExecutionBackend,
    KernelCandidate,
    TransferCapability,
)
from streamcompiler.ir.graph import HeterogeneousGraph, Instruction
from streamcompiler.ir.resource_graph import ComputeResource, ResourceGraph


class OpenCLBackend(ExecutionBackend):
    backend_id = "opencl"

    def available(self) -> bool:
        try:
            import pyopencl  # type: ignore  # noqa: F401

            return True
        except Exception:  # noqa: BLE001
            return False

    def discover_devices(self) -> ResourceGraph:
        graph = ResourceGraph(fingerprint="", backends_present=())
        if not self.available():
            graph.attributes["opencl_status"] = "runtime_unavailable"
            return graph
        graph = ResourceGraph(
            fingerprint="",
            backends_present=(self.backend_id,),
            attributes={"note": "OpenCL runtime importable; enumerate platforms before planning"},
        )
        return graph

    def supported_ops(self, device: ComputeResource) -> tuple[str, ...]:
        return device.supported_ops

    def supported_dtypes(self, device: ComputeResource) -> tuple[str, ...]:
        return device.supported_dtypes

    def enumerate_kernels(
        self, region: Instruction | HeterogeneousGraph, device: ComputeResource
    ) -> list[KernelCandidate]:
        return []

    def benchmark(self, candidate: KernelCandidate) -> BenchmarkResult:
        return BenchmarkResult(candidate, float("inf"), 0, False, "opencl not benchmarked")

    def compile(
        self, region: Instruction | HeterogeneousGraph, candidate: KernelCandidate
    ) -> CompiledRegion:
        raise RuntimeError("OpenCL backend compile requires a validated runtime path")

    def execute(self, executable: CompiledRegion, dependencies: Sequence[Any]) -> Any:
        raise RuntimeError("OpenCL backend execute requires a validated runtime path")

    def transfer_capabilities(
        self, source: ComputeResource | str, destination: ComputeResource | str
    ) -> TransferCapability:
        src = source if isinstance(source, str) else source.id.name
        dst = destination if isinstance(destination, str) else destination.id.name
        return TransferCapability(src, dst, kind="unsupported", notes="opencl transfers not validated")


class VulkanBackend(ExecutionBackend):
    backend_id = "vulkan"

    def available(self) -> bool:
        # Keep honest: presence of libvulkan is not enough without a compute path.
        return False

    def discover_devices(self) -> ResourceGraph:
        graph = ResourceGraph(
            fingerprint="",
            backends_present=(),
            attributes={"vulkan_status": "runtime_unavailable"},
        )
        return graph

    def supported_ops(self, device: ComputeResource) -> tuple[str, ...]:
        return ()

    def supported_dtypes(self, device: ComputeResource) -> tuple[str, ...]:
        return ()

    def enumerate_kernels(
        self, region: Instruction | HeterogeneousGraph, device: ComputeResource
    ) -> list[KernelCandidate]:
        return []

    def benchmark(self, candidate: KernelCandidate) -> BenchmarkResult:
        return BenchmarkResult(candidate, float("inf"), 0, False, "vulkan unavailable")

    def compile(
        self, region: Instruction | HeterogeneousGraph, candidate: KernelCandidate
    ) -> CompiledRegion:
        raise RuntimeError("Vulkan backend not available")

    def execute(self, executable: CompiledRegion, dependencies: Sequence[Any]) -> Any:
        raise RuntimeError("Vulkan backend not available")

    def transfer_capabilities(
        self, source: ComputeResource | str, destination: ComputeResource | str
    ) -> TransferCapability:
        src = source if isinstance(source, str) else source.id.name
        dst = destination if isinstance(destination, str) else destination.id.name
        return TransferCapability(src, dst, kind="unsupported", notes="vulkan unavailable")
