"""Tests for native-layer resource guardrails via tensortorrent._native."""

from __future__ import annotations

import pytest
import tensortorrent._native as native

_MiB = 1 << 20


# ---------------------------------------------------------------------------
# NativeCpuBackend.discover + budget_report
# ---------------------------------------------------------------------------


def test_native_cpu_backend_discover_budget_report() -> None:
    """discover() with explicit budget → budget_report() reflects that budget."""
    explicit_budget = 64 * _MiB
    backend = native.NativeCpuBackend.discover(
        compute_workers=1,
        io_workers=1,
        memory_budget_bytes=explicit_budget,
    )
    report = backend.budget_report()
    assert isinstance(report, dict)
    assert report["memory_budget_bytes"] == explicit_budget


def test_native_cpu_allocate_beyond_budget_raises() -> None:
    """Allocating more than the budget raises RuntimeError mentioning 'budget'."""
    budget = 64 * _MiB
    backend = native.NativeCpuBackend.discover(
        compute_workers=1,
        io_workers=1,
        memory_budget_bytes=budget,
    )
    with pytest.raises(RuntimeError, match="budget"):
        # Request way more than the budget
        backend.allocate("cpu", budget * 4, 64)


# ---------------------------------------------------------------------------
# NativeExecutionContext setters
# ---------------------------------------------------------------------------


def test_native_ctx_set_spill_budget_bytes() -> None:
    """set_spill_budget_bytes is callable on a fresh NativeExecutionContext."""
    ctx = native.NativeExecutionContext()
    ctx.set_spill_budget_bytes(200 * _MiB)  # must not raise


def test_native_ctx_set_stall_timeout_secs() -> None:
    """set_stall_timeout_secs is callable on a fresh NativeExecutionContext."""
    ctx = native.NativeExecutionContext()
    ctx.set_stall_timeout_secs(60.0)  # must not raise


def test_native_ctx_set_resource_capacity() -> None:
    """set_resource_capacity is callable on a fresh NativeExecutionContext."""
    ctx = native.NativeExecutionContext()
    ctx.set_resource_capacity("cpu", 512 * _MiB)  # must not raise


def test_native_ctx_execution_id_positive() -> None:
    """execution_id is a positive integer."""
    ctx = native.NativeExecutionContext()
    assert int(ctx.execution_id) > 0


def test_native_ctx_cancel_token_interaction() -> None:
    """NativeExecutionContext can be constructed with a NativeCancelToken."""
    token = native.NativeCancelToken()
    ctx = native.NativeExecutionContext(cancel_token=token)
    assert int(ctx.execution_id) > 0
