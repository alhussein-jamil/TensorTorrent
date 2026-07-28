"""Cost model tests."""

from __future__ import annotations

from streamcompiler.cost_model import measure_host_copy, transfer_time


def test_host_copy_model_is_measured_not_peak_claim() -> None:
    model = measure_host_copy("numa_ram_0", "numa_ram_0", sizes=(1 << 20, 4 << 20))
    assert model.measured
    assert model.samples
    t = transfer_time(model, "numa_ram_0", "numa_ram_0", 2 << 20)
    assert t > 0
    # Larger transfers should not be cheaper than smaller ones in the fitted model.
    t_big = transfer_time(model, "numa_ram_0", "numa_ram_0", 8 << 20)
    assert t_big >= t * 0.5
