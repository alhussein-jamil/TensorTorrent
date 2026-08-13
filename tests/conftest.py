"""Shared test fixtures.

Compiled artifacts, packs, and spill files must never land in the
developer's real ``~/.cache/tensortorrent`` (often read-only in this
environment). Unit, e2e, and hardware tests all honor ``TT_CACHE_DIR``.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session", autouse=True)
def _isolate_cache_and_spill_roots(tmp_path_factory: pytest.TempPathFactory) -> None:
    """Point the artifact cache and spill root at a session-scoped temp dir."""
    import os

    root = tmp_path_factory.mktemp("tt-cache")
    cache_dir = root / "cache"
    spill_dir = root / "spill"
    cache_dir.mkdir(parents=True, exist_ok=True)
    spill_dir.mkdir(parents=True, exist_ok=True)
    previous = {
        "TT_CACHE_DIR": os.environ.get("TT_CACHE_DIR"),
        "TT_SPILL_DIR": os.environ.get("TT_SPILL_DIR"),
        "TT_ALLOW_TMPFS_SPILL": os.environ.get("TT_ALLOW_TMPFS_SPILL"),
    }
    os.environ["TT_CACHE_DIR"] = str(cache_dir)
    os.environ["TT_SPILL_DIR"] = str(spill_dir)
    os.environ["TT_ALLOW_TMPFS_SPILL"] = "1"  # pytest tmp may be tmpfs
    yield
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
