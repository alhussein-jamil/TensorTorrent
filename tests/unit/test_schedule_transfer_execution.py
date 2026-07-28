"""GraphExecutor must run ExecutableSchedule Transfer ops before consumers."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import streamcompiler as sc
from streamcompiler.config import CompileConfig
from streamcompiler.ir.graph import OpCode
from streamcompiler.runtime.graph_executor import GraphExecutor
from streamcompiler.runtime.schedule import ExecutableSchedule, MemoryTier, PlanInstruction
from streamcompiler.runtime.tensor_store import ResidentParameterStore


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
    compiled = sc.compile(
        model,
        (x,),
        config=CompileConfig(
            use_torch_compile=False,
            max_concurrent_regions=2,
            max_region_nodes=2,
            measure_regions=False,
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
        instructions=list(base.instructions) + [transfer],
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
    assert executor.tensor_directory.has_copy_at(out_name, "cpu_numa_0_copy") or event.get("elided")
    compiled.close()
