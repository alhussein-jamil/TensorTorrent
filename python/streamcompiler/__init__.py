"""StreamCompiler: heterogeneous streaming compiler for PyTorch inference."""

from __future__ import annotations

from typing import Any

from streamcompiler.compile.pipeline import (
    PortableArtifact,
    SpecializedArtifact,
    portable_compile_from_ir,
    specialize_for_machine,
)
from streamcompiler.config import CompileConfig, Objective
from streamcompiler.errors import ExecutionCancelled, StreamCompilerError, UnsupportedFeatureError
from streamcompiler.frontend.export import compile as _compile
from streamcompiler.runtime.module import CompiledModule, load_compiled

__all__ = [
    "CompileConfig",
    "CompiledModule",
    "ExecutionCancelled",
    "Objective",
    "PortableArtifact",
    "SpecializedArtifact",
    "StreamCompilerError",
    "UnsupportedFeatureError",
    "compile",
    "load_compiled",
    "portable_compile_from_ir",
    "specialize_for_machine",
]

__version__ = "0.1.0"


def compile(
    model: Any,
    example_inputs: Any,
    config: CompileConfig | None = None,
    *,
    devices: str = "auto",
    machine: Any | None = None,
    measurements: Any | None = None,
    **kwargs: Any,
) -> CompiledModule:
    """Compile a PyTorch module into a machine-specialized ``torch.nn.Module``.

    ``example_inputs`` must be the positional arguments the model is called with,
    optionally as ``(args, kwargs)``. The returned module executes the real graph
    and returns outputs matching eager PyTorch.

    ``machine`` injects a :class:`~streamcompiler.ir.resource_graph.ResourceGraph`
    (for example a CPU + mock-accel graph in tests). ``measurements`` injects
    planner latencies for deterministic placement.
    """
    return _compile(
        model,
        example_inputs,
        config=config,
        devices=devices,
        machine=machine,
        measurements=measurements,
        **kwargs,
    )
