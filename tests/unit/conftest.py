"""Shared unit-test fixtures.

The important job here is isolation: compiled artifacts, packs, and spill
files must never be shared between tests, and must never land in the
developer's real ``~/.cache/tensortorrent``. Both locations are environment
overridable in production (read-only container roots need that too), so the
test suite uses the same public knobs rather than patching internals.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session", autouse=True)
def _isolate_cache_and_spill_roots(tmp_path_factory: pytest.TempPathFactory) -> None:
    """Point the artifact cache and spill root at a session-scoped temp dir.

    Without this, every test shares one on-disk cache keyed by fingerprint,
    so a cached artifact written by one test can be picked up by another and
    the developer's real cache accumulates test junk between runs.
    """
    import os

    root = tmp_path_factory.mktemp("tt-cache")
    cache_dir = root / "cache"
    spill_dir = root / "spill"
    cache_dir.mkdir(parents=True, exist_ok=True)
    spill_dir.mkdir(parents=True, exist_ok=True)
    previous = {
        "TT_CACHE_DIR": os.environ.get("TT_CACHE_DIR"),
        "TT_SPILL_DIR": os.environ.get("TT_SPILL_DIR"),
    }
    os.environ["TT_CACHE_DIR"] = str(cache_dir)
    os.environ["TT_SPILL_DIR"] = str(spill_dir)
    yield
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
