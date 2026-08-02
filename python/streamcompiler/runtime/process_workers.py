"""Optional process workers for concurrent region execution.

Production path uses Linux ``fork`` so region callables are inherited. The pool
keeps persistent workers, submits via a queue (nonblocking), and surfaces child
errors on the returned Future. This is not mixed-vendor CUDA process isolation.
"""

from __future__ import annotations

import contextlib
import itertools
import queue
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
        max_pending: int = 64,
    ) -> None:
        if isinstance(max_workers, bool) or not isinstance(max_workers, int) or max_workers < 1:
            raise RuntimePlanError("max_workers must be >= 1")
        self.max_workers = max_workers
        if isinstance(max_pending, bool) or not isinstance(max_pending, int) or max_pending < 1:
            raise RuntimePlanError("max_pending must be >= 1")
        self.max_pending = max_pending
        self.start_method = start_method
        self._ctx = mp.get_context(start_method)
        self._task_q: Any = self._ctx.Queue(maxsize=max_pending)
        self._result_q: Any = self._ctx.Queue(maxsize=max_pending)
        self._workers: list[Any] = []
        self._pending: dict[int, Future[Any]] = {}
        self._ids = itertools.count(1)
        self._lock = threading.Lock()
        self._closed = False
        self._broken_reason: str | None = None
        self._collector = threading.Thread(target=self._collect_results, name="sc-process-pool", daemon=True)
        try:
            for _ in range(self.max_workers):
                proc = self._ctx.Process(target=_worker_loop, args=(self._task_q, self._result_q), daemon=True)  # type: ignore[attr-defined]
                proc.start()
                self._workers.append(proc)
            self._collector.start()
            if warm_up:
                # Spawn/fork import cost is paid once so later submit() stays nonblocking.
                for _ in range(self.max_workers):
                    self.submit(_pool_ping).result(timeout=120)
        except BaseException:
            self.shutdown(wait=True)
            raise

    def _collect_results(self) -> None:
        while True:
            try:
                item = self._result_q.get(timeout=0.1)
            except queue.Empty:
                dead = [proc for proc in self._workers if not proc.is_alive()]
                if dead and not self._closed:
                    self._mark_broken(f"process worker exited unexpectedly (pids={[proc.pid for proc in dead]})")
                    break
                continue
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

    def _mark_broken(self, reason: str) -> None:
        with self._lock:
            self._broken_reason = reason
            pending = list(self._pending.values())
            self._pending.clear()
        for future in pending:
            if not future.done():
                future.set_exception(RuntimePlanError(reason))

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Future[Any]:
        """Queue work and return immediately. Does not wait for the child."""
        if self._closed:
            raise RuntimePlanError("ProcessWorkerPool is shut down")
        if self._broken_reason is not None:
            raise RuntimePlanError(f"ProcessWorkerPool is broken: {self._broken_reason}")
        fut: Future[Any] = Future()
        task_id = next(self._ids)
        with self._lock:
            if len(self._pending) >= self.max_pending:
                raise RuntimePlanError(f"backpressure: process worker pool at pending limit {self.max_pending}")
            self._pending[task_id] = fut
        try:
            self._task_q.put_nowait((task_id, fn, args, kwargs))
        except queue.Full as exc:
            with self._lock:
                self._pending.pop(task_id, None)
            raise RuntimePlanError("backpressure: process worker queue full") from exc
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
                self._task_q.put_nowait(None)
        for proc in self._workers:
            if wait:
                proc.join(timeout=timeout)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=timeout)
        with contextlib.suppress(Exception):
            self._result_q.put_nowait(None)
        if wait and self._collector.is_alive():
            self._collector.join(timeout=timeout)
        with self._lock:
            pending = list(self._pending.values())
            self._pending.clear()
        for fut in pending:
            if not fut.done():
                fut.set_exception(RuntimePlanError("ProcessWorkerPool shut down before task completed"))
        for queue_obj in (self._task_q, self._result_q):
            with contextlib.suppress(Exception):
                queue_obj.close()
            with contextlib.suppress(Exception):
                queue_obj.join_thread()
        self._workers.clear()
