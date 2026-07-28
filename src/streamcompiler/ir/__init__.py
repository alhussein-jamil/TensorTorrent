"""Intermediate representation packages."""

from streamcompiler.ir.graph import HeterogeneousGraph, Instruction, OpCode, TensorMeta
from streamcompiler.ir.resource_graph import (
    ComputeClass,
    ComputeResource,
    LinkClass,
    MemoryClass,
    MemoryResource,
    ResourceDecision,
    ResourceGraph,
    ResourceId,
    ResourceKind,
    TransferLink,
    ensure_host_staged_fallbacks,
    merge_graphs,
)

__all__ = [
    "ComputeClass",
    "ComputeResource",
    "HeterogeneousGraph",
    "Instruction",
    "LinkClass",
    "MemoryClass",
    "MemoryResource",
    "OpCode",
    "ResourceDecision",
    "ResourceGraph",
    "ResourceId",
    "ResourceKind",
    "TensorMeta",
    "TransferLink",
    "ensure_host_staged_fallbacks",
    "merge_graphs",
]
