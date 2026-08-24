#!/usr/bin/env python3
"""Print POSIX export lines for this host's native library path."""

from __future__ import annotations

import sys
import sysconfig
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tt_platform import load  # noqa: E402

print(load().emit_native_library_exports(sysconfig.get_config_var("LIBDIR") or ""))
