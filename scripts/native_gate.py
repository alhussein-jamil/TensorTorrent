"""CI/native path proof: public compile path is truly native."""

from __future__ import annotations

import torch
import torch.nn as nn

import streamcompiler as sc
from streamcompiler.config import CompileConfig
from streamcompiler.native import native_available, require_native
from streamcompiler.testing.native_oracle import (
    assert_native_runtime_used,
    assert_no_hot_path_schedule_conversion,
    assert_no_python_fallback,
)


def main() -> int:
    if not native_available():
        print("FAIL: native extension not loaded")
        return 1
    native = require_native()
    model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4)).eval()
    x = torch.randn(4, 8)
    compiled = sc.compile(
        model,
        (x,),
        config=CompileConfig(use_torch_compile=False, measure_regions=False),
    )
    try:
        # Counters after specialize — measure forward hot path only.
        native.reset_debug_counters()
        before = dict(native.debug_counters())
        out = compiled(x)
        torch.testing.assert_close(out, model(x))
        out2 = compiled(x)
        torch.testing.assert_close(out2, model(x))
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
        print(
            "PASS native-gate:",
            f"artifact_id={stats.get('native_artifact_id')}",
            f"execution_id={stats.get('native_execution_id')}",
            f"gil_delta={after.get('gil_acquisitions', 0) - before.get('gil_acquisitions', 0)}",
        )
        return 0
    finally:
        compiled.close()


if __name__ == "__main__":
    raise SystemExit(main())
