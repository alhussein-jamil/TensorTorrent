"""Testing-only Python DAG oracle. Never import from production forward path."""

from __future__ import annotations

from streamcompiler._legacy.dispatch import dispatch
from streamcompiler._legacy.runtime import run_schedule_legacy_python

__all__ = ["dispatch", "run_schedule_legacy_python"]
