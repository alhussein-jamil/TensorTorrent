"""Hardware validation suite behavior on the current machine."""

from __future__ import annotations

from tensortorrent.validation.hardware import CheckResult, CheckStatus, ValidationReport, validate_hardware


def test_validate_hardware_distinguishes_statuses() -> None:
    report = validate_hardware(full=False, stress=False)
    statuses = {c.status for c in report.checks}
    assert CheckStatus.HARDWARE_DETECTED in statuses
    assert CheckStatus.BACKEND_AVAILABLE in statuses
    # Missing accelerators must appear as unsupported/skipped, not silent success.
    text = report.render_text()
    assert "fingerprint:" in text
    summary = report.summary()
    assert "counts" in summary
    assert summary["fingerprint"] == report.fingerprint


def test_numerical_validation_executes_the_tensortorrent_path() -> None:
    """The numerics check must compile and run, not compare eager against eager."""
    report = validate_hardware(full=False, stress=False)
    check = next(c for c in report.checks if c.name == "numerical_equivalence_eager")
    assert check.status is CheckStatus.NUMERICAL_CORRECTNESS_VALIDATED
    assert "tensortorrent vs eager" in check.detail
    assert check.measured["region_count"] >= 1
    assert check.measured["wall_time_s"] > 0.0
    assert check.measured["max_abs_err"] < 1e-5


def test_dtype_capability_is_not_reported_as_compiled() -> None:
    """Listing a dtype is discovery, not evidence that a kernel was compiled."""
    report = validate_hardware(full=False, stress=False)
    dtype_checks = [c for c in report.checks if c.name.startswith("dtypes_reported:")]
    assert dtype_checks
    for check in dtype_checks:
        assert check.status is CheckStatus.HARDWARE_DETECTED
        assert "not compiled or executed" in check.detail
    assert not any(c.name.startswith("dtype:") for c in report.checks)


def test_absent_accelerators_are_never_reported_as_working() -> None:
    report = validate_hardware(full=False, stress=False)
    for check in report.checks:
        if check.name.startswith("backend_available:") and "not available" in check.detail:
            assert check.status is CheckStatus.UNSUPPORTED_CAPABILITY
    gpu_checks = [c for c in report.checks if c.name in ("concurrent_gpus", "concurrent_cpu_gpu")]
    import torch

    if not torch.cuda.is_available():
        for check in gpu_checks:
            assert check.status is CheckStatus.SKIPPED


def test_gpu_presence_reports_concurrent_topology() -> None:
    from tensortorrent.ir.resource_graph import (
        ComputeClass,
        ComputeResource,
        ResourceGraph,
        ResourceId,
        ResourceKind,
    )
    from tensortorrent.validation.hardware import ValidationReport, _validate_concurrency

    graph = ResourceGraph(fingerprint="gpu-presence")
    for i in range(2):
        graph.add_compute(
            ComputeResource(
                id=ResourceId(ResourceKind.COMPUTE, f"cuda_gpu_{i}"),
                compute_class=ComputeClass.DISCRETE_GPU,
                backend_id="cuda",
                model=f"g{i}",
                vendor="nvidia",
            )
        )
    graph.add_compute(
        ComputeResource(
            id=ResourceId(ResourceKind.COMPUTE, "cpu_numa_0"),
            compute_class=ComputeClass.CPU_NUMA_POOL,
            backend_id="cpu",
            model="cpu",
            vendor="cpu",
        )
    )
    # Without --full: topology is observed as skipped, never a production pass.
    report = ValidationReport(fingerprint="gpu-presence", started_unix=0.0)
    _validate_concurrency(report, graph, full=False)
    for name in ("concurrent_gpus", "concurrent_cpu_gpu"):
        check = next(c for c in report.checks if c.name == name)
        assert check.status is CheckStatus.SKIPPED
        assert check.measured.get("gpu_count") == 2
        assert "not" in check.detail.lower() or "validate-hardware --full" in check.detail


def test_production_ready_requires_measured_not_detection() -> None:
    report = ValidationReport(fingerprint="det-only", started_unix=0.0)
    report.add(
        CheckResult(
            name="discover_resource_graph",
            status=CheckStatus.HARDWARE_DETECTED,
            detail="present",
        )
    )
    report.add(
        CheckResult(
            name="backend_available:cuda",
            status=CheckStatus.BACKEND_AVAILABLE,
            detail="available",
        )
    )
    ready, blockers = report.production_ready()
    assert not ready
    assert any("basic_execution" in b for b in blockers)
    assert any("numerical_equivalence" in b for b in blockers)


def test_production_ready_on_current_host() -> None:
    report = validate_hardware(full=False, stress=False)
    ready, blockers = report.production_ready()
    summary = report.summary()
    assert summary["production_ready"] is ready
    assert summary["production_blockers"] == blockers
    if ready:
        assert not blockers
    else:
        assert blockers


def test_stress_runs_short_soak() -> None:
    report = validate_hardware(full=False, stress=True)
    soak = next(c for c in report.checks if c.name == "long_running_stability")
    assert soak.status is not CheckStatus.SKIPPED
    assert soak.measured.get("iters", 0) >= 30
    assert soak.status in {
        CheckStatus.BASIC_EXECUTION_VALIDATED,
        CheckStatus.FAILED,
    }


def test_single_gpu_does_not_claim_multi_gpu_readiness() -> None:
    from tensortorrent.ir.resource_graph import ComputeClass, ComputeResource, ResourceGraph, ResourceId, ResourceKind
    from tensortorrent.validation.hardware import ValidationReport, _validate_concurrency

    graph = ResourceGraph(fingerprint="single-gpu")
    graph.add_compute(
        ComputeResource(
            id=ResourceId(ResourceKind.COMPUTE, "cuda_gpu_0"),
            compute_class=ComputeClass.DISCRETE_GPU,
            backend_id="cuda",
            model="gpu",
            vendor="nvidia",
        )
    )
    report = ValidationReport(fingerprint="single-gpu", started_unix=0.0)
    _validate_concurrency(report, graph, full=True)

    check = next(c for c in report.checks if c.name == "concurrent_gpus")
    assert check.status is CheckStatus.SKIPPED
    assert check.measured["gpu_count"] == 1


def test_cpu_concurrency_claim_matches_the_measurement() -> None:
    report = validate_hardware(full=False, stress=False)
    check = next(c for c in report.checks if c.name == "concurrent_cpu_regions")
    if check.status is CheckStatus.CONCURRENT_EXECUTION_VALIDATED:
        assert check.measured["enabled"] is True
        assert "max_concurrent_regions=1" not in check.detail
    else:
        assert check.status is CheckStatus.SKIPPED
        assert check.measured["workers"] == 1
