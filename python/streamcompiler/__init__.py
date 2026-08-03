"""StreamCompiler: heterogeneous streaming compiler for PyTorch inference and opt-in schedule training."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from streamcompiler.compile.pipeline import (
    PortableArtifact,
    SpecializedArtifact,
    portable_compile_from_ir,
    specialize_for_machine,
)
from streamcompiler.config import CompileConfig, Objective
from streamcompiler.errors import ExecutionCancelled, StreamCompilerError, UnsupportedFeatureError
from streamcompiler.frontend.composition import GraphInput, ModuleGraph, ModuleNode, NodeOutput
from streamcompiler.frontend.export import capture_module, compile_exported
from streamcompiler.frontend.export import compile as _compile
from streamcompiler.runtime.module import CompiledModule, load_compiled
from streamcompiler.train import fit

__all__ = [
    "CompileConfig",
    "CompiledModule",
    "ExecutionCancelled",
    "GraphInput",
    "ModuleGraph",
    "ModuleNode",
    "NodeOutput",
    "Objective",
    "PortableArtifact",
    "SpecializedArtifact",
    "StreamCompilerError",
    "UnsupportedFeatureError",
    "capture_module",
    "compile",
    "compile_modules",
    "compile_exported",
    "fit",
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

    For large models, prefer :func:`capture_module` + :func:`compile_exported` so
    the eager module can be freed between export and specialization.

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


def compile_modules(
    modules: Sequence[Any],
    example_inputs: Any,
    *,
    names: Sequence[str] | None = None,
    config: CompileConfig | None = None,
    artifact_dir: str | Path | None = None,
    devices: str = "auto",
    machine: Any | None = None,
    measurements: Any | None = None,
) -> CompiledModule:
    """Compile multiple modules in series as one graph and executable artifact."""
    graph = ModuleGraph.series(modules, names=names)
    return compile(
        graph,
        example_inputs,
        config=config,
        artifact_dir=artifact_dir,
        devices=devices,
        machine=machine,
        measurements=measurements,
    )
