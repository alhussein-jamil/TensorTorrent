"""Unknown transfer paths must not stay labelled measured."""

from __future__ import annotations

from streamcompiler.planner.cost.transfer import TransferModel, transfer_time


def test_unknown_transfer_path_is_not_marked_measured() -> None:
    model = TransferModel(source="a", destination="b", measured=True)
    latency = model.predict(1 << 20)
    assert latency > 0
    assert model.measured is False


def test_measured_host_copy_stays_measured_after_predict() -> None:
    from streamcompiler.planner.cost import measure_host_copy

    model = measure_host_copy("a", "a", sizes=(1 << 20, 2 << 20))
    assert model.measured
    assert transfer_time(model, "a", "a", 1 << 20) > 0
    assert model.measured is True


def test_transfer_time_none_model_uses_prior() -> None:
    assert transfer_time(None, "a", "b", 1 << 20) > 0
