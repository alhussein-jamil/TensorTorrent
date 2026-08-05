"""CI/native path proof: public compile path is truly native."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

import tensortorrent as tt
from tensortorrent.config import CompileConfig
from tensortorrent.native import native_available, require_native

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tests.support.native import (  # noqa: E402
    assert_native_runtime_used,
    assert_no_hot_path_schedule_conversion,
    assert_no_python_fallback,
    assert_scheduler_entered,
    assert_zero_non_compute_callbacks,
)


def main() -> int:
    if not native_available():
        print("FAIL: native extension not loaded")
        return 1
    native = require_native()
    model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4)).eval()
    x = torch.randn(4, 8)
    compiled = tt.compile(
        model,
        (x,),
        config=CompileConfig(
            use_torch_compile=False,
            measure_regions=False,
            # Gate proves native schedule + artifact reuse; direct path skips that.
            prefer_direct_path=False,
        ),
    )
    try:
        expected = model(x)
        # Counters after specialize — measure forward hot path only.
        native.reset_debug_counters()
        before = dict(native.debug_counters())
        out = compiled(x)
        torch.testing.assert_close(out, expected, check_device=False)
        out2 = compiled(x)
        torch.testing.assert_close(out2, expected, check_device=False)
        report = compiled.last_report
        assert report is not None
        stats = report.parameter_store
        assert_native_runtime_used(stats)
        assert stats.get("native_data_plane") is True
        assert stats.get("native_artifact_reused") is True
        assert stats.get("native_execution_id") is not None
        after = dict(native.debug_counters())
        assert_no_hot_path_schedule_conversion(before, after)
        assert_no_python_fallback(before, after)
        assert_scheduler_entered(before, after, min_enters=2)
        assert_zero_non_compute_callbacks(before, after)
        assert after.get("compute_callbacks", 0) - before.get("compute_callbacks", 0) >= 2
        print(
            "PASS native-gate:",
            f"artifact_id={stats.get('native_artifact_id')}",
            f"execution_id={stats.get('native_execution_id')}",
            f"gil_delta={after.get('gil_acquisitions', 0) - before.get('gil_acquisitions', 0)}",
            f"non_compute={after.get('non_compute_python_callbacks', 0) - before.get('non_compute_python_callbacks', 0)}",
            f"compute={after.get('compute_callbacks', 0) - before.get('compute_callbacks', 0)}",
        )
        return 0
    finally:
        compiled.close()


if __name__ == "__main__":
    raise SystemExit(main())
