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
from streamcompiler.errors import StreamCompilerError, UnsupportedFeatureError
from streamcompiler.frontend.export import compile as _compile
from streamcompiler.runtime.module import CompiledModule, load_compiled

__all__ = [
    "CompileConfig",
    "CompiledModule",
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
    **kwargs: Any,
) -> CompiledModule:
    """Compile a PyTorch module into a machine-specialized ``torch.nn.Module``.

    ``example_inputs`` must be the positional arguments the model is called with,
    optionally as ``(args, kwargs)``. The returned module executes the real graph
    and returns outputs matching eager PyTorch.
    """
    return _compile(model, example_inputs, config=config, devices=devices, **kwargs)
