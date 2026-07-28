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
from streamcompiler.errors import StreamCompilerError
from streamcompiler.frontend.export import compile as _compile
from streamcompiler.runtime.module import CompiledModule

__all__ = [
    "CompileConfig",
    "CompiledModule",
    "Objective",
    "PortableArtifact",
    "SpecializedArtifact",
    "StreamCompilerError",
    "compile",
    "portable_compile_from_ir",
    "specialize_for_machine",
]

__version__ = "0.1.0"


def compile(model: Any, example_inputs: Any, config: CompileConfig | None = None, **kwargs: Any) -> CompiledModule:
    """Compile a PyTorch module into a machine-specialized executable wrapper."""
    return _compile(model, example_inputs, config=config, **kwargs)
