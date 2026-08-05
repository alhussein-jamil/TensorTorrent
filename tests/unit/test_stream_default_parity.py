"""Parity: Python stream defaults match Rust schedule_from_py defaults."""

from __future__ import annotations

from tensortorrent.ir.graph import OpCode
from tensortorrent.native import require_native
from tensortorrent.runtime.schedule import (
    ExecutableSchedule,
    PlanInstruction,
    default_stream_id,
    with_explicit_streams,
)


def test_default_stream_id_matches_native_fill() -> None:
    """Python default_stream_id must match what Rust fills on round-trip."""
    native = require_native()
    cases = [
        (OpCode.COMPUTE, "cpu_a", "cpu_a::compute"),
        (OpCode.TRANSFER, "copy_engine", "copy_engine::copy0"),
        (OpCode.LOAD, "cpu_a", "cpu_a::copy0"),
        (OpCode.PREFETCH, "cpu_a", "cpu_a::copy0"),
        (OpCode.RECORD_EVENT, "cpu_a", "cpu_a::sync"),
        (OpCode.WAIT_EVENT, "cpu_a", "cpu_a::sync"),
        (OpCode.RELEASE, "cpu_a", "cpu_a::lifetime"),
        (OpCode.EVICT, "cpu_a", "cpu_a::lifetime"),
    ]
    for opcode, resource, expected in cases:
        assert default_stream_id(opcode, resource) == expected

    # Build a schedule missing stream ids; after ensure_explicit_streams + native
    # validate (which converts via schedule_from_py defaults), no missing-stream errors.
    instructions = [
        PlanInstruction(
            opcode=OpCode.COMPUTE,
            name="c0",
            resource="cpu_a",
            executable_ref="c0",
            inputs=("x",),
            outputs=("y",),
        ),
        PlanInstruction(
            opcode=OpCode.TRANSFER,
            name="t0",
            resource="copy_engine",
            inputs=("y",),
            outputs=("y",),
            source="cpu_a",
            destination="cpu_b",
            depends_on=("c0",),
        ),
        PlanInstruction(
            opcode=OpCode.COMPUTE,
            name="c1",
            resource="cpu_b",
            executable_ref="c1",
            inputs=("y",),
            outputs=("z",),
            depends_on=("t0",),
        ),
    ]
    raw = ExecutableSchedule(graph_name="g", fingerprint="f", instructions=instructions)
    filled = ExecutableSchedule(
        graph_name="g",
        fingerprint="f",
        instructions=tuple(with_explicit_streams(i) for i in raw.instructions),
    )
    for inst in filled.instructions:
        assert inst.stream_id
        if inst.opcode in (OpCode.TRANSFER, OpCode.LOAD, OpCode.PREFETCH):
            assert inst.copy_engine_id
        if inst.opcode == OpCode.TRANSFER:
            assert inst.link_id
    assert native.validate_schedule(filled) == []
