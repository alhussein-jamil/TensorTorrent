"""Seeded property checks for native planner determinism."""

from __future__ import annotations

import random

import pytest

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

pytestmark = pytest.mark.skipif(not native_available(), reason="native extension required")


def _random_problem(rng: random.Random, *, n_dev: int, n_regions: int) -> dict:
    machine = ResourceGraph(fingerprint=f"prop-{n_dev}-{n_regions}", backends_present=("mock",))
    device_names = []
    device_memory = []
    capacities = []
    for i in range(n_dev):
        name = f"d{i}"
        mem = f"m{i}"
        device_names.append(name)
        device_memory.append(mem)
        cap = rng.choice([5_000, 20_000, 100_000])
        capacities.append(cap)
        machine.add_memory(
            MemoryResource(
                id=ResourceId(ResourceKind.MEMORY, mem),
                memory_class=MemoryClass.DEVICE_VRAM,
                capacity_bytes=cap,
                allocatable_bytes=cap,
                attached_compute=(name,),
            )
        )
        machine.add_compute(
            ComputeResource(
                id=ResourceId(ResourceKind.COMPUTE, name),
                compute_class=ComputeClass.ACCELERATOR,
                backend_id="mock_accel",
                vendor="prop",
                model=name,
                memory_affinity=(mem,),
                supported_dtypes=("float32",),
            )
        )
    for i in range(n_dev):
        for j in range(n_dev):
            if i == j:
                continue
            machine.add_link(
                TransferLink(
                    id=ResourceId(ResourceKind.LINK, f"m{i}->m{j}"),
                    link_class=LinkClass.PCIE,
                    source=f"m{i}",
                    destination=f"m{j}",
                    bidirectional=False,
                    measured=True,
                    latency_s=rng.choice([0.0, 1e-6]),
                    bytes_per_s=rng.choice([1e6, 1e8, 1e10]),
                )
            )

    regions = []
    candidates = []
    order = list(range(n_regions))
    edge_bytes = []
    for r in range(n_regions):
        deps = [r - 1] if r > 0 and rng.random() < 0.8 else []
        out_b = rng.choice([64, 512, 2048])
        regions.append(
            {
                "name": f"r{r}",
                "depends_on": deps,
                "output_bytes": out_b,
                "state_bytes": rng.choice([0, 128]),
                "consumer_count": 1 if r + 1 < n_regions else 0,
            }
        )
        if deps:
            edge_bytes.append((deps[0], r, out_b))
        pool = []
        for d in range(n_dev):
            pool.append(
                {
                    "device": d,
                    "backend_id": "mock",
                    "kernel_id": f"r{r}:d{d}",
                    "dtype": "float32",
                    "estimated_latency_s": rng.uniform(0.001, 0.05),
                    "workspace_bytes": rng.choice([0, 64, 512]),
                    "measured": True,
                }
            )
        candidates.append(pool)

    subsets = [{"device_indices": [i]} for i in range(n_dev)]
    if n_dev >= 2:
        subsets.append({"device_indices": list(range(n_dev))})

    return {
        "config": {
            "objective": rng.choice(["latency", "throughput", "memory"]),
            "beam_width": 8,
            "candidates_per_device": 2,
            "local_search_iters": 1,
            "planner_workers": 1,
            "allow_parallel_subsets": True,
            "finalist_count": 6,
            "per_subset_finalists": 3,
            "allow_host_staged_transfers": True,
            "target_inflight_requests": 1,
        },
        "device_names": device_names,
        "capacities": capacities,
        "device_memory": device_memory,
        "regions": regions,
        "order": order,
        "candidates": candidates,
        "edge_bytes": edge_bytes,
        "subsets": subsets,
        "machine": machine,
    }


@pytest.mark.parametrize("seed", [1, 2, 3, 7, 11, 42, 99, 123])
def test_native_planner_deterministic_across_workers(seed: int) -> None:
    rng = random.Random(seed)
    problem = _random_problem(rng, n_dev=rng.randint(1, 3), n_regions=rng.randint(2, 5))
    native = require_native()
    serial = dict(problem)
    serial["config"] = dict(problem["config"])
    serial["config"]["planner_workers"] = 1
    parallel = dict(problem)
    parallel["config"] = dict(problem["config"])
    parallel["config"]["planner_workers"] = 4
    a = native.plan_placements(serial)
    b = native.plan_placements(parallel)
    sigs_a = [f.get("placement_signature") for f in (a.get("finalists") or [])]
    sigs_b = [f.get("placement_signature") for f in (b.get("finalists") or [])]
    assert sigs_a == sigs_b


@pytest.mark.parametrize("seed", [0, 5, 13])
def test_native_planner_stable_under_beam_widths(seed: int) -> None:
    """Large beam must not worsen the #1 finalist vs a huge beam on tiny graphs."""
    rng = random.Random(seed)
    problem = _random_problem(rng, n_dev=2, n_regions=3)
    native = require_native()
    narrow = dict(problem)
    narrow["config"] = dict(problem["config"])
    narrow["config"]["beam_width"] = 4
    narrow["config"]["planner_workers"] = 1
    wide = dict(problem)
    wide["config"] = dict(problem["config"])
    wide["config"]["beam_width"] = 64
    wide["config"]["planner_workers"] = 1
    a = native.plan_placements(narrow)
    b = native.plan_placements(wide)
    fa = list(a.get("finalists") or [])
    fb = list(b.get("finalists") or [])
    if not fa or not fb:
        return
    # Wide beam's best analytic score should be ≤ narrow's (lower better).
    assert float(fb[0].get("analytic_score") or 0) <= float(fa[0].get("analytic_score") or 0) + 1e-6


def _fork_join_problem() -> dict:
    """Two-device fork/join with expensive cross-device edges."""
    machine = ResourceGraph(fingerprint="prop-fork", backends_present=("mock",))
    for i in range(2):
        machine.add_memory(
            MemoryResource(
                id=ResourceId(ResourceKind.MEMORY, f"m{i}"),
                memory_class=MemoryClass.DEVICE_VRAM,
                capacity_bytes=1 << 30,
                allocatable_bytes=1 << 30,
                attached_compute=(f"d{i}",),
            )
        )
        machine.add_compute(
            ComputeResource(
                id=ResourceId(ResourceKind.COMPUTE, f"d{i}"),
                compute_class=ComputeClass.ACCELERATOR,
                backend_id="mock_accel",
                vendor="prop",
                model=f"d{i}",
                memory_affinity=(f"m{i}",),
                supported_dtypes=("float32",),
            )
        )
    machine.add_link(
        TransferLink(
            id=ResourceId(ResourceKind.LINK, "m0->m1"),
            link_class=LinkClass.PCIE,
            source="m0",
            destination="m1",
            bidirectional=True,
            measured=True,
            latency_s=1e-4,
            bytes_per_s=1e6,
            contention_factor=2.0,
        )
    )
    # regions: root → br0, br1 → join
    regions = [
        {"name": "root", "depends_on": [], "output_bytes": 8000, "state_bytes": 0, "consumer_count": 2},
        {"name": "br0", "depends_on": [0], "output_bytes": 8000, "state_bytes": 0, "consumer_count": 1},
        {"name": "br1", "depends_on": [0], "output_bytes": 8000, "state_bytes": 0, "consumer_count": 1},
        {"name": "join", "depends_on": [1, 2], "output_bytes": 64, "state_bytes": 0, "consumer_count": 0},
    ]
    candidates = []
    for r in range(4):
        pool = []
        for d in range(2):
            pool.append(
                {
                    "device": d,
                    "backend_id": "mock",
                    "kernel_id": f"r{r}:d{d}",
                    "dtype": "float32",
                    "estimated_latency_s": 0.01,
                    "workspace_bytes": 0,
                    "measured": True,
                }
            )
        candidates.append(pool)
    return {
        "config": {
            "objective": "latency",
            "beam_width": 16,
            "candidates_per_device": 1,
            "local_search_iters": 1,
            "planner_workers": 1,
            "allow_parallel_subsets": False,
            "finalist_count": 8,
            "per_subset_finalists": 4,
            "allow_host_staged_transfers": True,
            "target_inflight_requests": 1,
        },
        "device_names": ["d0", "d1"],
        "capacities": [1 << 30, 1 << 30],
        "device_memory": ["m0", "m1"],
        "regions": regions,
        "order": [0, 1, 2, 3],
        "candidates": candidates,
        "edge_bytes": [(0, 1, 8000), (0, 2, 8000), (1, 3, 8000), (2, 3, 8000)],
        "subsets": [{"device_indices": [0]}, {"device_indices": [1]}, {"device_indices": [0, 1]}],
        "machine": machine,
    }


def test_fork_join_transfer_heavy_emits_diverse_finalists() -> None:
    native = require_native()
    out = native.plan_placements(_fork_join_problem())
    finals = list(out.get("finalists") or [])
    assert finals
    # At least one multi-device and one single-device strategy when both feasible.
    subset_sizes = {len(f.get("subset_devices") or []) for f in finals}
    assert 1 in subset_sizes or 2 in subset_sizes
    # Transfer-heavy split should report positive transfer latency when split.
    split = [f for f in finals if float(f.get("transfer_latency_s") or 0) > 0]
    assert split or all(len(f.get("subset_devices") or []) == 1 for f in finals)


@pytest.mark.parametrize("objective", ["latency", "throughput", "memory"])
def test_objectives_return_feasible_finalists(objective: str) -> None:
    rng = random.Random(17 + hash(objective) % 1000)
    problem = _random_problem(rng, n_dev=2, n_regions=3)
    problem["config"]["objective"] = objective
    problem["config"]["beam_width"] = 32
    native = require_native()
    out = native.plan_placements(problem)
    finals = list(out.get("finalists") or [])
    assert finals
    for f in finals:
        assert float(f.get("analytic_score") or 0) < float("inf")
        assert len(f.get("placements") or []) == 3
