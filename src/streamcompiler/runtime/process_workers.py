"""Optional process workers for mixed-vendor isolation.

Torch CUDA contexts are process-local; running incompatible vendor stacks in one
process is undefined. This pool keeps persistent spawned workers, submits via a
queue (nonblocking), and surfaces child errors on the returned Future.
"""

from __future__ import annotations

import contextlib
import itertools
import threading
from collections.abc import Callable
from concurrent.futures import Future
from typing import Any

import torch.multiprocessing as mp

from streamcompiler.errors import RuntimePlanError


def _worker_loop(task_q: Any, result_q: Any) -> None:
    while True:
        item = task_q.get()
        if item is None:
            break
        task_id, fn, args, kwargs = item
        try:
            result = fn(*args, **kwargs)
            result_q.put((task_id, "ok", result))
        except Exception as exc:  # noqa: BLE001 - boundary: surface child errors to parent
            result_q.put((task_id, "err", repr(exc)))


def _pool_ping() -> int:
    return 1


class ProcessWorkerPool:
    """Persistent process pool with nonblocking ``submit`` and clean shutdown."""

    def __init__(
        self,
        max_workers: int = 1,
        *,
        warm_up: bool = True,
        start_method: str = "spawn",
    ) -> None:
        self.max_workers = max(1, int(max_workers))
        self.start_method = start_method
        self._ctx = mp.get_context(start_method)
        self._task_q: Any = self._ctx.Queue()
        self._result_q: Any = self._ctx.Queue()
        self._workers: list[Any] = []
        self._pending: dict[int, Future[Any]] = {}
        self._ids = itertools.count(1)
        self._lock = threading.Lock()
        self._closed = False
        self._collector = threading.Thread(target=self._collect_results, name="sc-process-pool", daemon=True)
        for _ in range(self.max_workers):
            proc = self._ctx.Process(target=_worker_loop, args=(self._task_q, self._result_q), daemon=True)  # type: ignore[attr-defined]
            proc.start()
            self._workers.append(proc)
        self._collector.start()
        if warm_up:
            # Spawn/fork import cost is paid once so later submit() stays nonblocking.
            for _ in range(self.max_workers):
                self.submit(_pool_ping).result(timeout=120)

    def _collect_results(self) -> None:
        while True:
            try:
                item = self._result_q.get()
            except (EOFError, OSError):
                break
            if item is None:
                break
            task_id, status, payload = item
            with self._lock:
                fut = self._pending.pop(task_id, None)
            if fut is None or fut.cancelled():
                continue
            if status == "ok":
                fut.set_result(payload)
            else:
                fut.set_exception(RuntimePlanError(f"Process worker failed: {payload}"))

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Future[Any]:
        """Queue work and return immediately. Does not wait for the child."""
        if self._closed:
            raise RuntimePlanError("ProcessWorkerPool is shut down")
        fut: Future[Any] = Future()
        task_id = next(self._ids)
        with self._lock:
            self._pending[task_id] = fut
        try:
            self._task_q.put((task_id, fn, args, kwargs))
        except Exception:
            with self._lock:
                self._pending.pop(task_id, None)
            raise
        return fut

    def shutdown(self, *, wait: bool = True, timeout: float = 5.0) -> None:
        if self._closed:
            return
        self._closed = True
        for _ in self._workers:
            with contextlib.suppress(Exception):
                self._task_q.put(None)
        for proc in self._workers:
            if wait:
                proc.join(timeout=timeout)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=timeout)
        with contextlib.suppress(Exception):
            self._result_q.put(None)
        if wait and self._collector.is_alive():
            self._collector.join(timeout=timeout)
        with self._lock:
            pending = list(self._pending.values())
            self._pending.clear()
        for fut in pending:
            if not fut.done():
                fut.set_exception(RuntimePlanError("ProcessWorkerPool shut down before task completed"))
        self._workers.clear()
