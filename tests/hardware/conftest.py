"""Redirect scratch off tmpfs so oversize-model packs have room to spill."""

from __future__ import annotations

import gc
import os
from pathlib import Path

import pytest


@pytest.fixture(scope="session", autouse=True)
def _redirect_scratch_to_persistent_fs() -> None:
    scratch = Path(__file__).resolve().parents[2] / "target" / "tmp" / "hardware-tests"
    (scratch / "cache").mkdir(parents=True, exist_ok=True)
    (scratch / "spill").mkdir(parents=True, exist_ok=True)
    keys = ("TMPDIR", "TT_CACHE_DIR", "TT_SPILL_DIR", "TT_ALLOW_TMPFS_SPILL")
    previous = {k: os.environ.get(k) for k in keys}
    os.environ["TMPDIR"] = str(scratch)
    os.environ["TT_CACHE_DIR"] = str(scratch / "cache")
    os.environ["TT_SPILL_DIR"] = str(scratch / "spill")
    os.environ["TT_ALLOW_TMPFS_SPILL"] = "1"
    yield
    for k, v in previous.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


@pytest.fixture(autouse=True)
def _cuda_cleanup_between_hardware_tests() -> None:
    """Reclaim VRAM between cases so placement does not flake under suite load."""
    yield
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:  # noqa: BLE001
        pass
