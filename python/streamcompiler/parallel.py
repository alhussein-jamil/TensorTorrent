"""Thread-pool helpers for running region work off the calling thread.

PyTorch's ``inference_mode`` and grad-mode state are thread-local, so a worker
thread does not inherit the mode of the thread that submitted the work. Without an
initializer, regions dispatched to a pool would run with autograd recording on,
which both slows execution down and fails outright when the tensors involved were
created under ``inference_mode`` on the submitting thread.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import torch

_thread_state = threading.local()


def enter_inference_mode() -> None:
    """Put the calling thread into inference mode for the rest of its life.

    Used as a :class:`~concurrent.futures.ThreadPoolExecutor` initializer. The guard
    is kept alive in thread-local storage so it is released when the thread exits.
    """
    if getattr(_thread_state, "guard", None) is not None:
        return
    guard = torch.inference_mode(True)
    guard.__enter__()
    _thread_state.guard = guard


def inference_thread_pool(*, max_workers: int, thread_name_prefix: str) -> ThreadPoolExecutor:
    """Create a pool whose workers run in inference mode."""
    return ThreadPoolExecutor(
        max_workers=max_workers,
        thread_name_prefix=thread_name_prefix,
        initializer=enter_inference_mode,
    )
