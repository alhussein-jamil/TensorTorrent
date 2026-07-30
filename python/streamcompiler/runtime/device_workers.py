"""Per-device worker processes for accelerator isolation.

One process per device id. Coordinator keeps ownership of schedule; workers
own device-local work. Real CUDA contexts stay out until hardware exists —
this path is exercised with CPU callables and virtual device labels.

Not a substitute for in-process ``ProcessWorkerPool`` region offload; this is
the deployment-shaped supervisor (health, restart, capacity accounting hooks).
"""

from __future__ import annotations

import contextlib
import itertools
import threading
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any

import torch.multiprocessing as mp

from streamcompiler.errors import RuntimePlanError


def _worker_loop(device_id: str, task_q: Any, result_q: Any) -> None:
    while True:
        item = task_q.get()
        if item is None:
            break
        kind = item[0]
        if kind == "ping":
            result_q.put(("pong", device_id, item[1]))
            continue
        task_id, fn, args, kwargs = item[1], item[2], item[3], item[4]
        try:
            result = fn(*args, **kwargs)
            result_q.put(("ok", device_id, task_id, result))
        except Exception as exc:  # noqa: BLE001 — surface to coordinator
            result_q.put(("err", device_id, task_id, repr(exc)))


def run_region_on_device(
    call: Callable[..., Any],
    device: str,
    backend_id: str,
    region_id: str,
    args: tuple[Any, ...],
) -> tuple[dict[str, Any], tuple[Any, ...]]:
    """Top-level picklable region body for device worker processes."""
    import time

    import torch

    from streamcompiler.backends.torch_device import coerce_region_result

    start = time.perf_counter()
    if torch.is_inference_mode_enabled():
        result = call(*args)
    else:
        with torch.inference_mode():
            result = call(*args)
    outputs = coerce_region_result(result)
    end = time.perf_counter()
    return (
        {
            "region_id": region_id,
            "device": device,
            "backend_id": backend_id,
            "start_s": start,
            "end_s": end,
            "worker": f"device-{device}",
        },
        outputs,
    )


@dataclass
class DeviceWorkerStatus:
    device_id: str
    alive: bool
    pid: int | None
    restarts: int
    pending: int


@dataclass
class DeviceWorkerSupervisor:
    """One isolated process per ``device_id`` with health checks and restart."""

    device_ids: list[str]
    start_method: str = "spawn"
    _ctx: Any = field(init=False, repr=False)
    _task_qs: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _result_q: Any = field(init=False, repr=False)
    _procs: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _restarts: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _pending: dict[int, tuple[str, Future[Any]]] = field(default_factory=dict, init=False, repr=False)
    _pongs: dict[int, threading.Event] = field(default_factory=dict, init=False, repr=False)
    _ids: Any = field(init=False, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _collector: threading.Thread | None = field(default=None, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.device_ids:
            raise RuntimePlanError("DeviceWorkerSupervisor requires at least one device_id")
        if len(set(self.device_ids)) != len(self.device_ids):
            raise RuntimePlanError("DeviceWorkerSupervisor device_ids must be unique")
        self._ctx = mp.get_context(self.start_method)
        self._result_q = self._ctx.Queue()
        self._ids = itertools.count(1)
        for did in self.device_ids:
            self._restarts[did] = 0
            self._spawn(did)
        self._collector = threading.Thread(target=self._collect, name="sc-device-workers", daemon=True)
        self._collector.start()

    def _spawn(self, device_id: str) -> None:
        task_q = self._ctx.Queue()
        proc = self._ctx.Process(
            target=_worker_loop,
            args=(device_id, task_q, self._result_q),
            daemon=True,
            name=f"sc-device-{device_id}",
        )
        proc.start()
        self._task_qs[device_id] = task_q
        self._procs[device_id] = proc

    def _collect(self) -> None:
        while True:
            try:
                item = self._result_q.get()
            except (EOFError, OSError):
                break
            if item is None:
                break
            kind = item[0]
            if kind == "pong":
                token = int(item[2])
                with self._lock:
                    ev = self._pongs.pop(token, None)
                if ev is not None:
                    ev.set()
                continue
            _, _device_id, task_id, payload = item
            with self._lock:
                entry = self._pending.pop(int(task_id), None)
            if entry is None:
                continue
            _did, fut = entry
            if fut.cancelled():
                continue
            if kind == "ok":
                fut.set_result(payload)
            else:
                fut.set_exception(RuntimePlanError(f"device worker failed: {payload}"))

    def health(self) -> list[DeviceWorkerStatus]:
        with self._lock:
            pending_by_dev: dict[str, int] = {d: 0 for d in self.device_ids}
            for did, _fut in self._pending.values():
                pending_by_dev[did] = pending_by_dev.get(did, 0) + 1
            out: list[DeviceWorkerStatus] = []
            for did in self.device_ids:
                proc = self._procs.get(did)
                alive = bool(proc is not None and proc.is_alive())
                out.append(
                    DeviceWorkerStatus(
                        device_id=did,
                        alive=alive,
                        pid=int(proc.pid) if proc is not None and proc.pid is not None else None,
                        restarts=self._restarts.get(did, 0),
                        pending=pending_by_dev.get(did, 0),
                    )
                )
            return out

    def ensure_healthy(self) -> list[str]:
        """Restart any dead workers. Returns restarted device ids."""
        restarted: list[str] = []
        with self._lock:
            if self._closed:
                raise RuntimePlanError("DeviceWorkerSupervisor is shut down")
            for did in self.device_ids:
                proc = self._procs.get(did)
                if proc is not None and proc.is_alive():
                    continue
                if proc is not None:
                    with contextlib.suppress(Exception):
                        proc.terminate()
                        proc.join(timeout=1.0)
                self._restarts[did] = self._restarts.get(did, 0) + 1
                self._spawn(did)
                restarted.append(did)
        return restarted

    def ping(self, device_id: str, *, timeout_s: float = 5.0) -> bool:
        if device_id not in self._task_qs:
            raise RuntimePlanError(f"unknown device_id {device_id}")
        token = next(self._ids)
        ev = threading.Event()
        with self._lock:
            self._pongs[token] = ev
        try:
            self._task_qs[device_id].put(("ping", token))
        except Exception:
            with self._lock:
                self._pongs.pop(token, None)
            raise
        ok = ev.wait(timeout=timeout_s)
        with self._lock:
            self._pongs.pop(token, None)
        return ok

    def submit(self, device_id: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Future[Any]:
        if self._closed:
            raise RuntimePlanError("DeviceWorkerSupervisor is shut down")
        if device_id not in self._task_qs:
            raise RuntimePlanError(f"unknown device_id {device_id}")
        self.ensure_healthy()
        fut: Future[Any] = Future()
        task_id = next(self._ids)
        with self._lock:
            self._pending[task_id] = (device_id, fut)
        try:
            self._task_qs[device_id].put(("task", task_id, fn, args, kwargs))
        except Exception:
            with self._lock:
                self._pending.pop(task_id, None)
            raise
        return fut

    def shutdown(self, *, wait: bool = True, timeout: float = 5.0) -> None:
        if self._closed:
            return
        self._closed = True
        for did, task_q in list(self._task_qs.items()):
            with contextlib.suppress(Exception):
                task_q.put(None)
            proc = self._procs.get(did)
            if proc is None:
                continue
            if wait:
                proc.join(timeout=timeout)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=timeout)
        with contextlib.suppress(Exception):
            self._result_q.put(None)
        if wait and self._collector is not None and self._collector.is_alive():
            self._collector.join(timeout=timeout)
        with self._lock:
            pending = list(self._pending.values())
            self._pending.clear()
        for _did, fut in pending:
            if not fut.done():
                fut.set_exception(RuntimePlanError("device worker shut down before task completed"))
        self._procs.clear()
        self._task_qs.clear()
