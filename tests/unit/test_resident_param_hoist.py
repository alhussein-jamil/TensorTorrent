"""Resident GPU parameter hoist: warm forwards must not re-upload weights."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import tensortorrent as tt
from tensortorrent.ir.graph import OpCode
from tensortorrent.native import native_available

pytestmark = pytest.mark.skipif(not native_available(), reason="native required")


class _TinyCudaLinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_resident_gpu_params_hoisted_from_steady_state_schedule() -> None:
    model = _TinyCudaLinear().eval()
    x = torch.randn(2, 8)
    compiled = tt.compile(
        model,
        (x,),
        config=tt.CompileConfig(
            use_torch_compile=False,
            measure_regions=True,
            region_measure_iters=1,
            prefer_direct_path=False,
            allow_concurrent_regions=False,
            allow_cpu=False,
            validate_numerics=False,
        ),
    )
    try:
        se = compiled.executor._schedule_executor
        assert se is not None
        devices = set(compiled.specialized.plan.devices_used)
        assert any(d.startswith("cuda_gpu_") for d in devices)
        canonical = [
            i
            for i in se.schedule.instructions
            if i.opcode == OpCode.TRANSFER and str(i.attributes.get("kind") or "") == "parameter_host_to_device"
        ]
        canonical_evicts = [
            i
            for i in se.schedule.instructions
            if i.opcode == OpCode.EVICT and str(i.attributes.get("kind") or "") == "parameter_evict"
        ]
        assert canonical, "expected parameter H2D transfers in the explain schedule"
        assert canonical_evicts, "expected parameter_evict in the explain schedule"
        runtime_names = set(se._native_instruction_names)
        assert all(inst.name not in runtime_names for inst in canonical)
        assert all(inst.name not in runtime_names for inst in canonical_evicts)
        assert se._resident_parameter_targets

        with torch.inference_mode():
            out0 = compiled(x)
            out1 = compiled(x)
        torch.testing.assert_close(out0, out1, atol=1e-5, rtol=1e-5)
        assert se._persistent_device_param_cache
    finally:
        compiled.close()


def test_compute_region_dependencies_follow_schedule_edges() -> None:
    from tensortorrent.runtime.direct_path import _compute_region_dependencies
    from tensortorrent.runtime.schedule import ExecutableSchedule, PlanInstruction

    schedule = ExecutableSchedule(
        graph_name="g",
        fingerprint="f",
        instructions=(
            PlanInstruction(OpCode.COMPUTE, "compute::region_0", "cuda_gpu_0", executable_ref="region_0"),
            PlanInstruction(
                OpCode.TRANSFER,
                "transfer::a",
                "cpu_numa_0",
                depends_on=("compute::region_0",),
                inputs=("a",),
                outputs=("a",),
            ),
            PlanInstruction(
                OpCode.WAIT_EVENT,
                "wait::a",
                "cpu_numa_0",
                depends_on=("transfer::a",),
            ),
            PlanInstruction(
                OpCode.COMPUTE,
                "compute::region_1",
                "cpu_numa_0",
                depends_on=("wait::a",),
                executable_ref="region_1",
            ),
        ),
    )
    deps = _compute_region_dependencies(schedule)
    assert deps == {"region_0": set(), "region_1": {"region_0"}}
