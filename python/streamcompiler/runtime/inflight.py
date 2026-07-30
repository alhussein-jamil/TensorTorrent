"""Allow N concurrent forwards; close waits for drain."""

from __future__ import annotations

import threading


class InFlightGate:
    """Non-exclusive gate: many runners, close waits until idle."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._inflight = 0
        self._closed = False

    @property
    def closed(self) -> bool:
        with self._cond:
            return self._closed

    @property
    def inflight(self) -> int:
        with self._cond:
            return self._inflight

    def enter(self) -> None:
        with self._cond:
            if self._closed:
                raise RuntimeError("gate closed")
            self._inflight += 1

    def leave(self) -> None:
        with self._cond:
            self._inflight = max(0, self._inflight - 1)
            self._cond.notify_all()

    def mark_closed_and_wait(self) -> None:
        with self._cond:
            self._closed = True
            while self._inflight > 0:
                self._cond.wait()

    def wait_idle(self) -> None:
        """Block until no runners (gate stays open)."""
        with self._cond:
            while self._inflight > 0:
                self._cond.wait()

    def assert_open(self) -> None:
        with self._cond:
            if self._closed:
                raise RuntimeError("gate closed")
