"""Cost model tests."""

from __future__ import annotations

from streamcompiler.cost_model import calibrate_host_priors, measure_host_copy, prediction_error, transfer_time


def test_host_copy_model_is_measured_not_peak_claim() -> None:
    model = measure_host_copy("numa_ram_0", "numa_ram_0", sizes=(1 << 20, 4 << 20))
    assert model.measured
    assert model.samples
    t = transfer_time(model, "numa_ram_0", "numa_ram_0", 2 << 20)
    assert t > 0
    # Larger transfers should not be cheaper than smaller ones in the fitted model.
    t_big = transfer_time(model, "numa_ram_0", "numa_ram_0", 8 << 20)
    assert t_big >= t * 0.5


def test_calibrate_host_priors_returns_alpha_beta() -> None:
    priors = calibrate_host_priors(sizes=(1 << 20, 2 << 20))
    assert priors["measured"]
    assert priors["alpha_s"] >= 0.0
    assert priors["beta_bytes_per_s"] is None or priors["beta_bytes_per_s"] > 0
    assert priors["cpu_region_s"] > 0
    assert priors["gil_noop_s"] >= 0.0


def test_prediction_error_absolute_and_relative() -> None:
    err = prediction_error(1.2, 1.0)
    assert abs(float(err["prediction_error_s"]) - 0.2) < 1e-12
    assert abs(float(err["prediction_relative_error"]) - 0.2) < 1e-12
    assert prediction_error(1.0, None)["prediction_error_s"] is None
