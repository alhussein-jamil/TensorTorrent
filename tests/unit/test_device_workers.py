"""Device worker supervisor: isolation, health, restart."""

from __future__ import annotations

import os
import signal
import time
from typing import Any

import pytest

from streamcompiler.errors import RuntimePlanError
from streamcompiler.runtime.device_workers import DeviceWorkerSupervisor
from streamcompiler.serve import InferenceService


def _mul(a: int, b: int) -> int:
    return a * b


def _sleep(seconds: float) -> None:
    time.sleep(seconds)


def _add_tensors(a: Any, b: Any) -> Any:
    return a + b


def test_device_worker_submit_and_ping() -> None:
    sup = DeviceWorkerSupervisor(device_ids=["virtual_0", "virtual_1"])
    try:
        assert all(s.alive for s in sup.health())
        assert sup.ping("virtual_0")
        assert sup.submit("virtual_1", _mul, 6, 7).result(timeout=30) == 42
    finally:
        sup.shutdown()


def test_device_worker_restarts_after_kill() -> None:
    sup = DeviceWorkerSupervisor(device_ids=["virtual_0"])
    try:
        before = {s.device_id: s for s in sup.health()}
        pid = before["virtual_0"].pid
        assert pid is not None
        os.kill(pid, signal.SIGKILL)
        # Wait until the OS reaps the child.
        deadline = time.time() + 5.0
        while time.time() < deadline:
            proc = sup._procs.get("virtual_0")
            if proc is None or not proc.is_alive():
                break
            time.sleep(0.05)
        restarted = sup.ensure_healthy()
        assert restarted == ["virtual_0"]
        status = {s.device_id: s for s in sup.health()}
        assert status["virtual_0"].alive
        assert status["virtual_0"].restarts >= 1
        assert status["virtual_0"].pid != pid
        assert sup.submit("virtual_0", _mul, 2, 3).result(timeout=30) == 6
    finally:
        sup.shutdown()


def test_worker_crash_fails_its_pending_futures() -> None:
    sup = DeviceWorkerSupervisor(device_ids=["virtual_0"])
    try:
        future = sup.submit("virtual_0", _sleep, 30.0)
        pid = sup.health()[0].pid
        assert pid is not None
        os.kill(pid, signal.SIGKILL)
        with pytest.raises(RuntimePlanError, match="crashed"):
            future.result(timeout=5)
        assert sup.health()[0].restarts == 1
    finally:
        sup.shutdown()


def test_device_worker_backpressure_is_bounded() -> None:
    sup = DeviceWorkerSupervisor(device_ids=["virtual_0"], max_pending_per_device=1)
    try:
        first = sup.submit("virtual_0", _sleep, 0.2)
        with pytest.raises(RuntimePlanError, match="backpressure"):
            sup.submit("virtual_0", _mul, 2, 3)
        first.result(timeout=5)
    finally:
        sup.shutdown()


def test_device_worker_restart_budget_is_bounded() -> None:
    sup = DeviceWorkerSupervisor(device_ids=["virtual_0"], max_restarts=1)
    try:
        first_pid = sup.health()[0].pid
        assert first_pid is not None
        os.kill(first_pid, signal.SIGKILL)
        deadline = time.time() + 5
        while sup.health()[0].alive and time.time() < deadline:
            time.sleep(0.01)
        assert sup.ensure_healthy() == ["virtual_0"]
        second_pid = sup.health()[0].pid
        assert second_pid is not None
        os.kill(second_pid, signal.SIGKILL)
        deadline = time.time() + 5
        while sup.health()[0].alive and time.time() < deadline:
            time.sleep(0.01)
        with pytest.raises(RuntimePlanError, match="restart limit"):
            sup.ensure_healthy()
    finally:
        sup.shutdown()


def test_device_worker_rejects_duplicate_ids() -> None:
    with pytest.raises(RuntimePlanError, match="unique"):
        DeviceWorkerSupervisor(device_ids=["a", "a"])


def test_device_worker_rejects_empty_ids() -> None:
    with pytest.raises(RuntimePlanError, match="non-empty strings"):
        DeviceWorkerSupervisor(device_ids=[""])


@pytest.mark.parametrize(
    "kwargs",
    ({"max_pending_per_device": 1.5}, {"max_restarts": 1.5}),
)
def test_device_worker_rejects_non_integer_limits(kwargs: dict[str, Any]) -> None:
    with pytest.raises(RuntimePlanError):
        DeviceWorkerSupervisor(device_ids=["a"], **kwargs)


def test_service_health_reports_device_workers() -> None:
    workers = DeviceWorkerSupervisor(device_ids=["virtual_0"])
    svc = InferenceService(device_workers=workers)
    svc.start()
    try:
        health = svc.health()
        assert health["device_workers"] is not None
        assert health["device_workers"][0]["device_id"] == "virtual_0"
        assert health["device_workers"][0]["alive"] is True
    finally:
        svc.stop()


def test_run_region_on_device_via_supervisor() -> None:
    import torch

    from streamcompiler.runtime.device_workers import run_region_on_device

    sup = DeviceWorkerSupervisor(device_ids=["cpu_worker"], start_method="fork")
    try:
        a = torch.ones(2)
        b = torch.ones(2)
        meta, outs = sup.submit(
            "cpu_worker",
            run_region_on_device,
            _add_tensors,
            "cpu_worker",
            "cpu",
            "r0",
            (a, b),
        ).result(timeout=30)
        assert meta["region_id"] == "r0"
        assert meta["worker"] == "device-cpu_worker"
        assert len(outs) == 1
        assert torch.equal(outs[0], torch.full((2,), 2.0))
    finally:
        sup.shutdown()
