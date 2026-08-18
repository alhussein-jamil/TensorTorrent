"""Load ``tensortorrent.platform`` without executing the torch-heavy package init."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

_ROOT = Path(__file__).resolve().parents[1]
_PLATFORM_PATH = _ROOT / "python" / "tensortorrent" / "platform.py"


def load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("tt_host_platform", _PLATFORM_PATH)
    if spec is None or spec.loader is None:
        raise SystemExit(f"missing platform module: {_PLATFORM_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
