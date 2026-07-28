"""Improve redundancy pass: drop duplicate consecutive identical transfers."""

from __future__ import annotations

from dataclasses import dataclass, field

from streamcompiler.ir.graph import HeterogeneousGraph, OpCode


@dataclass
class RedundancyReport:
    eliminated_transfers: list[str] = field(default_factory=list)
    eliminated_conversions: list[str] = field(default_factory=list)


def eliminate_redundancy(graph: HeterogeneousGraph) -> RedundancyReport:
    report = RedundancyReport()
    kept = []
    last_transfer_key: tuple[str, str, str] | None = None
    for inst in graph.instructions:
        if inst.opcode in (OpCode.TRANSFER, OpCode.MATERIALIZE):
            key = (inst.opcode.value, inst.source or "", inst.destination or "")
            if last_transfer_key == key:
                report.eliminated_transfers.append(inst.name)
                continue
            last_transfer_key = key
        else:
            last_transfer_key = None
        if inst.opcode in (OpCode.COMPRESS, OpCode.DECOMPRESS):
            # Track only; real conversion CSE needs residency analysis.
            pass
        kept.append(inst)
    graph.instructions = kept
    return report
