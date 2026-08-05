"""Fingerprint helpers shared by compiled artifacts and the runtime."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensortorrent.compile.pipeline import SpecializedArtifact
    from tensortorrent.ir.resource_graph import ResourceGraph


def specialized_fingerprint_mismatch(artifact: SpecializedArtifact, machine: ResourceGraph) -> bool:
    """True when a cached artifact was specialized for a different machine."""
    mismatched = bool(artifact.fingerprint and machine.fingerprint and artifact.fingerprint != machine.fingerprint)
    if mismatched:
        from tensortorrent.backends.torch_device import clear_compile_cache

        clear_compile_cache()
    return mismatched
