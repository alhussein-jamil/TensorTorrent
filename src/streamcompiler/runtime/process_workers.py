"""Optional process workers for mixed-vendor isolation.

Torch CUDA contexts are process-local; running incompatible vendor stacks in one
process is undefined. This pool runs callables in spawned children with tensor
IPC via ``torch.multiprocessing``.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future
from typing import Any

import torch.multiprocessing as mp

from streamcompiler.errors import RuntimePlanError


def _child_entry(conn: Any, fn: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
    try:
        result = fn(*args, **kwargs)
        conn.send(("ok", result))
    except Exception as exc:  # noqa: BLE001 - boundary: surface child errors to parent
        conn.send(("err", repr(exc)))
    finally:
        conn.close()


class ProcessWorkerPool:
    """Small process pool; prefer threads unless vendor isolation is required."""

    def __init__(self, max_workers: int = 1) -> None:
        self.max_workers = max(1, int(max_workers))
        self._ctx = mp.get_context("spawn")
        self._active: list[Any] = []

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Future[Any]:
        if len(self._active) >= self.max_workers:
            raise RuntimePlanError("ProcessWorkerPool is at capacity; wait for a worker to finish")
        parent, child = self._ctx.Pipe(duplex=False)
        proc = self._ctx.Process(target=_child_entry, args=(child, fn, args, kwargs))
        fut: Future[Any] = Future()
        proc.start()
        child.close()
        self._active.append(proc)

        def _wait() -> Any:
            status, payload = parent.recv()
            proc.join(timeout=60)
            parent.close()
            if proc in self._active:
                self._active.remove(proc)
            if status == "ok":
                return payload
            raise RuntimePlanError(f"Process worker failed: {payload}")

        # Run wait in-thread for simplicity; callers that need async should use threads.
        try:
            fut.set_result(_wait())
        except Exception as exc:
            fut.set_exception(exc)
        return fut

    def shutdown(self) -> None:
        for proc in list(self._active):
            if proc.is_alive():
                proc.terminate()
            proc.join(timeout=5)
        self._active.clear()
