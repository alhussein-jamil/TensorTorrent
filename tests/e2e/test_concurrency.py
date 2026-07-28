"""Concurrent CPU region execution: real overlap, and never across dependencies."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import streamcompiler as sc
from streamcompiler.compile.concurrency import (
    dependency_levels,
    measure_concurrency_benefit,
    transitive_dependencies,
)
from streamcompiler.compile.measure import capture_region_inputs


class TwoBranches(nn.Module):
    """Two independent, deliberately heavy branches that join at the end."""

    def __init__(self, width: int = 512) -> None:
        super().__init__()
        self.stem = nn.Linear(width, width)
        self.left = nn.Linear(width, width)
        self.right = nn.Linear(width, width)
        self.head = nn.Linear(width, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.stem(x))
        return self.head(torch.relu(self.left(h)) + torch.tanh(self.right(h)))


class Chain(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(32, 32)
        self.b = nn.Linear(32, 32)
        self.c = nn.Linear(32, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c(torch.relu(self.b(torch.relu(self.a(x)))))


def test_independent_regions_actually_overlap() -> None:
    model = TwoBranches().eval()
    x = torch.randn(96, 512)
    compiled = sc.compile(model, (x,), config=sc.CompileConfig(max_concurrent_regions=2))
    with torch.no_grad():
        expected = model(x)

    overlaps: list[tuple[str, str]] = []
    for _ in range(8):
        actual = compiled(x)
        report = compiled.last_report
        assert report is not None
        overlaps.extend(report.overlapping_pairs())
        if overlaps:
            break
    torch.testing.assert_close(actual, expected)
    assert overlaps, "independent regions never overlapped in time"


def test_overlapping_regions_are_never_dependent() -> None:
    """The scheduler must not run a region while its ancestor is still running."""
    model = TwoBranches(width=256).eval()
    x = torch.randn(64, 256)
    compiled = sc.compile(model, (x,), config=sc.CompileConfig(max_concurrent_regions=4))
    ancestors = transitive_dependencies(compiled.program)
    for _ in range(6):
        compiled(x)
        report = compiled.last_report
        assert report is not None
        for first, second in report.overlapping_pairs():
            assert second not in ancestors[first], f"{second} ran during its dependent {first}"
            assert first not in ancestors[second], f"{first} ran during its dependent {second}"


def test_chain_never_overlaps() -> None:
    """A purely sequential graph must stay sequential even with workers available."""
    model = Chain().eval()
    x = torch.randn(8, 32)
    compiled = sc.compile(model, (x,), config=sc.CompileConfig(max_concurrent_regions=4))
    for _ in range(5):
        compiled(x)
        report = compiled.last_report
        assert report is not None
        assert report.overlapping_pairs() == []


def test_dependency_levels_group_only_independent_regions() -> None:
    model = TwoBranches(width=64).eval()
    compiled = sc.compile(model, (torch.randn(4, 64),), config=sc.CompileConfig(max_concurrent_regions=2))
    levels = dependency_levels(compiled.program)
    ancestors = transitive_dependencies(compiled.program)
    assert levels.width >= 2, "branching model must expose independent regions"
    for level in levels.levels:
        for a in level:
            for b in level:
                if a != b:
                    assert b not in ancestors[a]


def test_concurrency_decision_is_measured_not_assumed() -> None:
    model = TwoBranches(width=256).eval()
    x = torch.randn(64, 256)
    compiled = sc.compile(model, (x,))
    decision = compiled.specialized.validation["concurrency"]
    assert decision["measured"] is True
    assert decision["sequential_s"] > 0.0
    assert decision["parallel_s"] > 0.0
    assert len(decision["group"]) >= 2
    # Whatever the verdict, it must be justified by the numbers it reports.
    if decision["enabled"]:
        assert decision["speedup"] >= 1.0
        assert decision["workers"] >= 2
    else:
        assert decision["workers"] == 1


def test_concurrency_is_skipped_for_sequential_graphs() -> None:
    compiled = sc.compile(Chain().eval(), (torch.randn(4, 32),))
    decision = compiled.specialized.validation["concurrency"]
    assert decision["enabled"] is False
    assert decision["workers"] == 1


def test_measure_concurrency_respects_single_worker_budget() -> None:
    model = TwoBranches(width=64).eval()
    x = torch.randn(4, 64)
    compiled = sc.compile(model, (x,))
    program = compiled.program
    flat = program.flatten_inputs((x,), {})
    inputs = capture_region_inputs(program, flat)
    decision = measure_concurrency_benefit(program, inputs, max_workers=1)
    assert decision.enabled is False
    assert decision.measured is False
    assert "one worker" in decision.reason


def test_concurrent_execution_matches_sequential_execution() -> None:
    model = TwoBranches(width=128).eval()
    x = torch.randn(32, 128)
    sequential = sc.compile(model, (x,), config=sc.CompileConfig(allow_concurrent_regions=False))
    concurrent = sc.compile(model, (x,), config=sc.CompileConfig(max_concurrent_regions=3))
    assert sequential._executor.max_workers == 1
    assert concurrent._executor.max_workers == 3
    torch.testing.assert_close(concurrent(x), sequential(x))


class ManyBranches(nn.Module):
    """Enough independent width that dividing the cores can pay off."""

    def __init__(self, width: int = 1024, branches: int = 8) -> None:
        super().__init__()
        self.stem = nn.Linear(width, width)
        self.branches = nn.ModuleList([nn.Linear(width, width) for _ in range(branches)])
        self.head = nn.Linear(width, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.stem(x))
        acc = torch.zeros_like(h)
        for branch in self.branches:
            acc = acc + torch.relu(branch(h))
        return self.head(acc)


def test_concurrency_measurement_divides_intra_op_threads() -> None:
    """Workers that each claim every core only contend, so splits must be measured."""
    model = ManyBranches().eval()
    x = torch.randn(64, 1024)
    compiled = sc.compile(model, (x,))
    decision = compiled.specialized.validation["concurrency"]
    reason = decision["reason"]
    assert reason.count("t=") >= 2, f"expected several thread splits to be timed: {reason}"
    if decision["enabled"]:
        # A measured win must come with the thread split that produced it.
        assert decision["intraop_threads"] >= 1
        assert compiled.executor.intraop_threads == decision["intraop_threads"]
        assert decision["parallel_s"] < decision["sequential_s"]
        assert "full-graph" in reason, "enabled concurrency must be confirmed on the full DAG"
    else:
        assert decision["intraop_threads"] == 0
        assert compiled.executor.intraop_threads == 0
    with torch.no_grad():
        torch.testing.assert_close(compiled(x), model(x))


def test_concurrent_execution_restores_the_process_thread_count() -> None:
    model = TwoBranches(width=256).eval()
    x = torch.randn(32, 256)
    compiled = sc.compile(model, (x,), config=sc.CompileConfig(max_concurrent_regions=2))
    compiled.executor.intraop_threads = 2
    before = torch.get_num_threads()
    compiled(x)
    assert torch.get_num_threads() == before


def test_thread_split_is_applied_only_while_regions_overlap(monkeypatch: pytest.MonkeyPatch) -> None:
    model = TwoBranches(width=128).eval()
    x = torch.randn(16, 128)
    compiled = sc.compile(model, (x,), config=sc.CompileConfig(max_concurrent_regions=2))
    compiled.executor.intraop_threads = 3

    calls: list[int] = []
    real = torch.set_num_threads
    monkeypatch.setattr(torch, "set_num_threads", lambda n: (calls.append(n), real(n))[1])
    compiled(x)
    assert calls[0] == 3, "the measured split must be applied for the call"
    assert calls[-1] == torch.get_num_threads(), "the process setting must be restored"
