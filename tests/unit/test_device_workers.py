"""Device worker supervisor: isolation, health, restart."""

from __future__ import annotations

import os
import signal
import time

import pytest
from server import InferenceService

from streamcompiler.errors import RuntimePlanError
from streamcompiler.runtime.device_workers import DeviceWorkerSupervisor


def _mul(a: int, b: int) -> int:
    return a * b


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


def test_device_worker_rejects_duplicate_ids() -> None:
    with pytest.raises(RuntimePlanError, match="unique"):
        DeviceWorkerSupervisor(device_ids=["a", "a"])


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
