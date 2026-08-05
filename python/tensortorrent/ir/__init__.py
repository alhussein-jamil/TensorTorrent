"""Intermediate representation and graph analysis."""

from tensortorrent.ir.alias import AliasAnalysis, run_alias_analysis
from tensortorrent.ir.graph import HeterogeneousGraph, Instruction, OpCode, TensorMeta
from tensortorrent.ir.liveness import (
    LivenessAnalysis,
    ranges_overlap,
    run_liveness_analysis,
)
from tensortorrent.ir.repeated_blocks import detect_repeated_blocks
from tensortorrent.ir.resource_graph import (
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
    "AliasAnalysis",
    "ComputeClass",
    "ComputeResource",
    "HeterogeneousGraph",
    "Instruction",
    "LinkClass",
    "LivenessAnalysis",
    "MemoryClass",
    "MemoryResource",
    "OpCode",
    "ResourceDecision",
    "ResourceGraph",
    "ResourceId",
    "ResourceKind",
    "TensorMeta",
    "TransferLink",
    "detect_repeated_blocks",
    "ensure_host_staged_fallbacks",
    "merge_graphs",
    "ranges_overlap",
    "run_alias_analysis",
    "run_liveness_analysis",
]
