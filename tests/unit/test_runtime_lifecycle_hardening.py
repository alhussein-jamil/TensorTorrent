"""Production lifecycle ownership regressions."""

from __future__ import annotations

import pytest

from tensortorrent.errors import TensorTorrentError
from tensortorrent.serve.model_manager import ModelManager


class _Ledger:
    def __init__(self) -> None:
        self.acquires = 0
        self.releases = 0

    def max_concurrent(self) -> int:
        return 4

    def acquire_or_raise(self, **_: object) -> None:
        self.acquires += 1

    def release(self) -> None:
        self.releases += 1


class _Module:
    def __init__(self) -> None:
        self.capacity_ledger = _Ledger()
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_model_manager_does_not_own_capacity_leases() -> None:
    module = _Module()
    manager = ModelManager()
    manager.load("m", module, concurrency_limit=4)  # type: ignore[arg-type]
    first = manager.acquire("m")
    second = manager.acquire("m")
    assert module.capacity_ledger.acquires == 0
    manager.release_slot(first)
    manager.release_slot(second)
    assert module.capacity_ledger.releases == 0


def test_model_manager_double_release_is_idempotent() -> None:
    manager = ModelManager()
    manager.load("m", _Module(), concurrency_limit=2)  # type: ignore[arg-type]
    slot = manager.acquire("m")
    manager.release_slot(slot)
    manager.release_slot(slot)
    assert slot.in_flight == 0


class _ZeroLedger(_Ledger):
    def max_concurrent(self) -> int:
        return 0


def test_model_manager_rejects_unserviceable_capacity() -> None:
    module = _Module()
    module.capacity_ledger = _ZeroLedger()
    manager = ModelManager()
    with pytest.raises(TensorTorrentError, match="cannot admit one request"):
        manager.load("m", module, concurrency_limit=2)  # type: ignore[arg-type]
