"""GraphExecutor must run ExecutableSchedule Transfer ops before consumers."""

from __future__ import annotations

import dataclasses

import pytest
import torch
import torch.nn as nn

import tensortorrent as tt
from tensortorrent.config import CompileConfig
from tensortorrent.ir.graph import OpCode
from tensortorrent.runtime.graph_executor import GraphExecutor
from tensortorrent.runtime.schedule import ExecutableSchedule, MemoryTier, PlanInstruction
from tensortorrent.runtime.tensor_store import ResidentParameterStore


def _order_release_after(instructions: list[PlanInstruction], tensor: str, dependency: str) -> list[PlanInstruction]:
    """Make ``release::<tensor>`` depend on ``dependency``.

    A Transfer appended to an already-built schedule is invisible to the
    Release instruction that the scheduler generated for the same tensor:
    ``release::<tensor>`` only depends on the record/wait events that existed
    at build time. Nothing then orders the appended Transfer before the
    Release, so the scheduler is free to free the tensor first and the
    Transfer fails with "not resident". Injecting the edge expresses the
    intent ("transfer between producer and consumer") in the DAG itself,
    which is what the executor actually schedules from.
    """
    out: list[PlanInstruction] = []
    for inst in instructions:
        if inst.opcode == OpCode.RELEASE and tensor in inst.inputs:
            inst = dataclasses.replace(inst, depends_on=tuple(dict.fromkeys((*inst.depends_on, dependency))))
        out.append(inst)
    return out


class Branching(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Linear(16, 16)
        self.left = nn.Linear(16, 16)
        self.right = nn.Linear(16, 16)
        self.head = nn.Linear(16, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.stem(x))
        return self.head(torch.relu(self.left(h)) + torch.tanh(self.right(h)))


def test_executor_runs_host_memcpy_transfer_from_schedule() -> None:
    model = Branching().eval()
    x = torch.randn(2, 16)
    compiled = tt.compile(
        model,
        (x,),
        config=CompileConfig(
            use_torch_compile=False,
            max_concurrent_regions=2,
            max_region_nodes=2,
            measure_regions=False,
            allow_gpu=False,
        ),
    )
    if len(compiled.regions) < 2:
        pytest.skip("fusion collapsed to one region; need multi-region for transfer injection")
    program = compiled.program
    producer = program.regions[0]
    consumer = program.regions[1]
    out_name = producer.outputs[0]
    base = compiled.specialized.schedule
    assert base is not None
    transfer = PlanInstruction(
        opcode=OpCode.TRANSFER,
        name=f"transfer::{producer.region_id}->{consumer.region_id}:test",
        resource="copy_engine:test",
        depends_on=(f"compute::{producer.region_id}",),
        inputs=(out_name,),
        outputs=(out_name,),
        nbytes=64,
        memory_tier=MemoryTier.SYSTEM_RAM,
        source="cpu_numa_0",
        destination="cpu_numa_0_copy",
        transfer_backend="host_memcpy",
        sync_required=True,
        attributes={"before_region": consumer.region_id, "after_region": producer.region_id},
    )
    schedule = ExecutableSchedule(
        graph_name=base.graph_name,
        fingerprint=base.fingerprint,
        # The release of ``out_name`` must wait for the injected transfer;
        # without that edge the two are unordered and race (flaky "not
        # resident" failures under concurrent dispatch).
        instructions=_order_release_after(list(base.instructions) + [transfer], out_name, transfer.name),
        notes=list(base.notes),
    )
    executor = GraphExecutor(
        program,
        compiled.executor.bindings,
        parameter_store=ResidentParameterStore(program.state_tensors()),
        max_workers=1,
        schedule=schedule,
    )
    with torch.no_grad():
        flat, _report = executor.run(program.flatten_inputs((x,), {}))
        torch.testing.assert_close(program.unflatten_outputs(flat), model(x))
    assert executor._transfer_events, "scheduled host memcpy must leave a transfer event"
    event = executor._transfer_events[0]
    assert event["simulated"] is False
    sreport = executor._last_schedule_report
    assert sreport is not None
    snap = sreport.copy_snapshot
    assert any(k.startswith(f"{out_name}@") and "copy" in k for k in snap) or event.get("elided")
    compiled.close()
