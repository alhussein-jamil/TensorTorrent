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
