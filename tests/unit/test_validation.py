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
    assert check.measured["max_abs_err"] == 0.0


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


def test_cpu_concurrency_claim_matches_the_measurement() -> None:
    report = validate_hardware(full=False, stress=False)
    check = next(c for c in report.checks if c.name == "concurrent_cpu_regions")
    if check.status is CheckStatus.CONCURRENT_EXECUTION_VALIDATED:
        assert check.measured["enabled"] is True
        assert "max_concurrent_regions=1" not in check.detail
    else:
        assert check.status is CheckStatus.SKIPPED
        assert check.measured["workers"] == 1
