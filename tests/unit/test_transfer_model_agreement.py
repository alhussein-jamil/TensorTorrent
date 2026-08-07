"""Planner estimate, DES transfer event, and MachineModel duration must agree."""

from __future__ import annotations

import pytest

from tensortorrent.ir.graph import OpCode
from tensortorrent.ir.resource_graph import (
    ComputeClass,
    ComputeResource,
    LinkClass,
    MemoryClass,
    MemoryResource,
    ResourceGraph,
    ResourceId,
    ResourceKind,
    TransferLink,
)
from tensortorrent.native import native_available, require_native
from tensortorrent.runtime.schedule import ExecutableSchedule, PlanInstruction
from tensortorrent.runtime.simulator.discrete_event import simulate_schedule

pytestmark = pytest.mark.skipif(not native_available(), reason="native required")


def _machine(*, contention: float = 1.0, bidirectional: bool = False) -> ResourceGraph:
    g = ResourceGraph(fingerprint="xfer-agree", backends_present=("mock",))
    for i, name in enumerate(("a0", "a1")):
        mem = f"v{i}"
        g.add_memory(
            MemoryResource(
                id=ResourceId(ResourceKind.MEMORY, mem),
                memory_class=MemoryClass.DEVICE_VRAM,
                capacity_bytes=1 << 30,
                allocatable_bytes=1 << 30,
                attached_compute=(name,),
            )
        )
        g.add_compute(
            ComputeResource(
                id=ResourceId(ResourceKind.COMPUTE, name),
                compute_class=ComputeClass.ACCELERATOR,
                backend_id="virtual",
                vendor="t",
                model=name,
                memory_affinity=(mem,),
                supported_dtypes=("float32",),
            )
        )
    g.add_link(
        TransferLink(
            id=ResourceId(ResourceKind.LINK, "v0->v1"),
            link_class=LinkClass.PCIE,
            source="v0",
            destination="v1",
            bidirectional=bidirectional,
            measured=True,
            latency_s=1e-4,
            bytes_per_s=1e6,
            contention_factor=contention,
        )
    )
    if not bidirectional:
        g.add_link(
            TransferLink(
                id=ResourceId(ResourceKind.LINK, "v1->v0"),
                link_class=LinkClass.PCIE,
                source="v1",
                destination="v0",
                bidirectional=False,
                measured=True,
                latency_s=2e-4,
                bytes_per_s=5e5,
                contention_factor=1.0,
            )
        )
    return g


def test_planner_and_des_agree_on_transfer_duration() -> None:
    native = require_native()
    machine = _machine(contention=2.0)
    nbytes = 2000
    # Manual: (lat + n/bw) * contention = (1e-4 + 2000/1e6) * 2 = 0.0042
    expected = (1e-4 + nbytes / 1e6) * 2.0

    problem = {
        "config": {
            "objective": "latency",
            "beam_width": 8,
            "candidates_per_device": 1,
            "local_search_iters": 0,
            "planner_workers": 1,
            "allow_parallel_subsets": False,
            "finalist_count": 4,
            "per_subset_finalists": 2,
            "allow_host_staged_transfers": True,
            "target_inflight_requests": 1,
        },
        "device_names": ["a0", "a1"],
        "capacities": [1 << 30, 1 << 30],
        "device_memory": ["v0", "v1"],
        "regions": [
            {"name": "r0", "depends_on": [], "output_bytes": nbytes, "state_bytes": 0, "consumer_count": 1},
            {"name": "r1", "depends_on": [0], "output_bytes": 4, "state_bytes": 0, "consumer_count": 0},
        ],
        "order": [0, 1],
        "candidates": [
            [
                {
                    "device": 0,
                    "backend_id": "m",
                    "kernel_id": "r0:a0",
                    "dtype": "float32",
                    "estimated_latency_s": 0.001,
                    "workspace_bytes": 0,
                    "measured": True,
                }
            ],
            [
                {
                    "device": 1,
                    "backend_id": "m",
                    "kernel_id": "r1:a1",
                    "dtype": "float32",
                    "estimated_latency_s": 0.001,
                    "workspace_bytes": 0,
                    "measured": True,
                }
            ],
        ],
        "edge_bytes": [(0, 1, nbytes)],
        "subsets": [{"device_indices": [0, 1]}],
        "machine": machine,
    }
    out = native.plan_placements(problem)
    finalists = list(out["finalists"])
    assert finalists
    split = next(f for f in finalists if f["placements"][0]["device"] != f["placements"][1]["device"])
    assert abs(float(split["transfer_latency_s"]) - expected) < 1e-9, split["transfer_latency_s"]

    # DES schedule with explicit Transfer of same nbytes/direction.
    sched = ExecutableSchedule(
        graph_name="g",
        fingerprint="fp",
        instructions=(
            PlanInstruction(
                opcode=OpCode.TRANSFER,
                name="t0",
                resource="copy_engine:v0->v1",
                inputs=("x",),
                outputs=("x",),
                nbytes=nbytes,
                source="v0",
                destination="v1",
            ),
        ),
    )
    sim = simulate_schedule(sched, machine)
    # Makespan equals transfer duration when only one transfer.
    assert abs(sim.makespan_s - expected) < 1e-9, sim.makespan_s
    assert sim.transfer_events or sim.timeline
    # Timeline/resource busy should reflect contention-scaled duration.
    busy = sum(sim.resource_busy_s.values())
    assert busy + 1e-12 >= expected


def test_directional_links_disagree_forward_reverse() -> None:
    machine = _machine(contention=1.0, bidirectional=False)
    nbytes = 1000
    fwd = (1e-4 + nbytes / 1e6) * 1.0
    rev = (2e-4 + nbytes / 5e5) * 1.0
    assert fwd != rev
    native = require_native()
    # Probe via plan with forced placements using only direction costs in transfer_latency.
    for src_dev, dst_dev, expect in ((0, 1, fwd), (1, 0, rev)):
        problem = {
            "config": {
                "objective": "latency",
                "beam_width": 4,
                "candidates_per_device": 1,
                "local_search_iters": 0,
                "planner_workers": 1,
                "allow_parallel_subsets": False,
                "finalist_count": 2,
                "per_subset_finalists": 1,
                "allow_host_staged_transfers": False,
                "target_inflight_requests": 1,
            },
            "device_names": ["a0", "a1"],
            "capacities": [1 << 30, 1 << 30],
            "device_memory": ["v0", "v1"],
            "regions": [
                {
                    "name": "r0",
                    "depends_on": [],
                    "output_bytes": nbytes,
                    "state_bytes": 0,
                    "consumer_count": 1,
                },
                {
                    "name": "r1",
                    "depends_on": [0],
                    "output_bytes": 4,
                    "state_bytes": 0,
                    "consumer_count": 0,
                },
            ],
            "order": [0, 1],
            "candidates": [
                [
                    {
                        "device": src_dev,
                        "backend_id": "m",
                        "kernel_id": "r0",
                        "dtype": "float32",
                        "estimated_latency_s": 1e-6,
                        "workspace_bytes": 0,
                        "measured": True,
                    }
                ],
                [
                    {
                        "device": dst_dev,
                        "backend_id": "m",
                        "kernel_id": "r1",
                        "dtype": "float32",
                        "estimated_latency_s": 1e-6,
                        "workspace_bytes": 0,
                        "measured": True,
                    }
                ],
            ],
            "edge_bytes": [(0, 1, nbytes)],
            "subsets": [{"device_indices": [0, 1]}],
            "machine": machine,
        }
        out = native.plan_placements(problem)
        f = out["finalists"][0]
        assert abs(float(f["transfer_latency_s"]) - expect) < 1e-9, (
            src_dev,
            dst_dev,
            f["transfer_latency_s"],
            expect,
        )
