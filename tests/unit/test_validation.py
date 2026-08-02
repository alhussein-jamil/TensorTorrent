"""Hardware validation suite behavior on the current machine."""

from __future__ import annotations

from streamcompiler.validation.hardware import CheckStatus, validate_hardware


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


def test_numerical_validation_executes_the_streamcompiler_path() -> None:
    """The numerics check must compile and run, not compare eager against eager."""
    report = validate_hardware(full=False, stress=False)
    check = next(c for c in report.checks if c.name == "numerical_equivalence_eager")
    assert check.status is CheckStatus.NUMERICAL_CORRECTNESS_VALIDATED
    assert "streamcompiler vs eager" in check.detail
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
    from streamcompiler.ir.resource_graph import (
        ComputeClass,
        ComputeResource,
        ResourceGraph,
        ResourceId,
        ResourceKind,
    )
    from streamcompiler.validation.hardware import ValidationReport, _validate_concurrency

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
    report = ValidationReport(fingerprint="gpu-presence", started_unix=0.0)
    _validate_concurrency(report, graph, full=True)
    for name in ("concurrent_gpus", "concurrent_cpu_gpu"):
        check = next(c for c in report.checks if c.name == name)
        assert check.status is CheckStatus.HARDWARE_DETECTED
        assert check.measured.get("gpu_count") == 2


def test_single_gpu_does_not_claim_multi_gpu_readiness() -> None:
    from streamcompiler.ir.resource_graph import ComputeClass, ComputeResource, ResourceGraph, ResourceId, ResourceKind
    from streamcompiler.validation.hardware import ValidationReport, _validate_concurrency

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
