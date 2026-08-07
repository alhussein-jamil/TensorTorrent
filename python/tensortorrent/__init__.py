"""TensorTorrent: heterogeneous streaming compiler for PyTorch inference and opt-in schedule training."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from tensortorrent._compat import require_torch
from tensortorrent.compile.pipeline import (
    PortableArtifact,
    SpecializedArtifact,
    portable_compile_from_ir,
    specialize_for_machine,
)
from tensortorrent.config import CompileConfig, Objective
from tensortorrent.errors import ExecutionCancelled, TensorTorrentError, UnsupportedFeatureError
from tensortorrent.frontend.composition import GraphInput, ModuleGraph, ModuleNode, NodeOutput
from tensortorrent.frontend.export import capture_module, compile_exported
from tensortorrent.frontend.export import compile as _compile
from tensortorrent.runtime.module import CompiledModule, load_compiled
from tensortorrent.train import fit

require_torch()

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
    "TensorTorrentError",
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

__version__ = "0.2.7"


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

    ``machine`` injects a :class:`~tensortorrent.ir.resource_graph.ResourceGraph`
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
