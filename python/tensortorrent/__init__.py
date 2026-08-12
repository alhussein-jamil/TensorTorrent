"""TensorTorrent: heterogeneous streaming compiler for PyTorch inference and opt-in schedule training."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from tensortorrent._compat import require_torch
from tensortorrent.closed import DeviceSelection, DeviceSelectionStr, NumericalMode, ProfileLevel
from tensortorrent.compile.pipeline import (
    PortableArtifact,
    SpecializedArtifact,
    portable_compile_from_ir,
    specialize_for_machine,
)
from tensortorrent.config import CompileConfig, Objective
from tensortorrent.errors import ExecutionCancelled, TensorTorrentError, UnsupportedFeatureError
from tensortorrent.frontend.composition import GraphInput, ModuleGraph, ModuleNode, NodeOutput
from tensortorrent.frontend.export import capture_module, compile_exported, load_exported_program
from tensortorrent.frontend.export import compile as _compile
from tensortorrent.runtime.module import CompiledModule, load_compiled
from tensortorrent.train import fit

require_torch()

__all__ = [
    "CompileConfig",
    "CompiledModule",
    "DeviceSelection",
    "ExecutionCancelled",
    "GraphInput",
    "ModuleGraph",
    "ModuleNode",
    "NodeOutput",
    "NumericalMode",
    "Objective",
    "PortableArtifact",
    "ProfileLevel",
    "SpecializedArtifact",
    "TensorTorrentError",
    "UnsupportedFeatureError",
    "capture_module",
    "compile",
    "compile_modules",
    "compile_exported",
    "fit",
    "load_compiled",
    "load_exported_program",
    "portable_compile_from_ir",
    "specialize_for_machine",
]

__version__ = "0.3.3"


def compile(
    model: Any,
    example_inputs: Any,
    config: CompileConfig | None = None,
    *,
    devices: DeviceSelection | DeviceSelectionStr = DeviceSelection.AUTO,
    machine: Any | None = None,
    measurements: Any | None = None,
    **kwargs: Any,
) -> CompiledModule:
    """Compile a PyTorch module into a machine-specialized ``torch.nn.Module``.

    ``example_inputs`` are the positional args the model is called with,
    optionally as ``(args, kwargs)``. Output matches eager PyTorch.

    Large models: prefer :func:`capture_module` + :func:`compile_exported` so
    the eager module can be freed between export and specialization.

    ``machine`` injects a :class:`~tensortorrent.ir.resource_graph.ResourceGraph`
    (e.g. CPU + mock-accel in tests). ``measurements`` injects planner
    latencies for deterministic placement.
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
    devices: DeviceSelection | DeviceSelectionStr = DeviceSelection.AUTO,
    machine: Any | None = None,
    measurements: Any | None = None,
) -> CompiledModule:
    """Compile modules in series as one graph / artifact."""
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
