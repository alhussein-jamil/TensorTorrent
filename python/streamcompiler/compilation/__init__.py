"""Specialize and AOT compile for a machine."""

from __future__ import annotations

from streamcompiler.compile.pipeline import (
    PortableArtifact,
    SpecializedArtifact,
    portable_compile_from_ir,
    specialize_for_machine,
)

__all__ = [
    "PortableArtifact",
    "SpecializedArtifact",
    "portable_compile_from_ir",
    "specialize_for_machine",
]
