"""Fork-worker region dispatch shared by GraphExecutor and ScheduleExecutor.

Lives outside both modules so neither needs to import the other for process
workers — breaks the ``graph_executor ↔ schedule_executor`` cycle.
"""

from __future__ import annotations

import itertools
import os
import time
from dataclasses import dataclass
from typing import Any

from tensortorrent.backends.torch_device import coerce_region_result

# Fork workers inherit this table; keyed by executor instance id.
_FORK_REGION_CALLABLES: dict[int, dict[str, Any]] = {}
_FORK_EXECUTOR_IDS = itertools.count(1)


@dataclass
class RegionEvent:
    region_id: str
    device: str
    backend_id: str
    start_s: float
    end_s: float
    worker: str

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def next_fork_registry_id() -> int:
    return next(_FORK_EXECUTOR_IDS)


def register_fork_callables(registry_id: int, callables: dict[str, Any]) -> None:
    _FORK_REGION_CALLABLES[registry_id] = dict(callables)


def unregister_fork_callables(registry_id: int) -> None:
    _FORK_REGION_CALLABLES.pop(registry_id, None)


def fork_run_region(
    registry_id: int,
    region_id: str,
    device: str,
    backend_id: str,
    args: tuple[Any, ...],
) -> tuple[RegionEvent, tuple[Any, ...]]:
    """Entry point submitted to :class:`ProcessWorkerPool` workers."""
    start = time.perf_counter()
    call = _FORK_REGION_CALLABLES[registry_id][region_id]
    result = call(*args)
    outputs = coerce_region_result(result)
    end = time.perf_counter()
    return (
        RegionEvent(
            region_id=region_id,
            device=device,
            backend_id=backend_id,
            start_s=start,
            end_s=end,
            worker=f"proc-{os.getpid()}",
        ),
        outputs,
    )
