"""Child-process GPU eager probe (isolates CUDA OOM from the parent suite)."""

from __future__ import annotations

import json
import sys

import torch
import torch.nn as nn


class DeepMLP(nn.Module):
    """Same layout as ``benchmarks.workloads.DeepMLP`` (kept local for script launch)."""

    def __init__(self, width: int, depth: int, out_features: int = 8) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(width, width) for _ in range(depth)])
        self.head = nn.Linear(width, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = torch.relu(blk(x))
        return self.head(x)


def main() -> int:
    payload = json.loads(sys.stdin.read())
    width = int(payload["width"])
    depth = int(payload["depth"])
    batch = int(payload["batch"])
    torch.manual_seed(0)
    try:
        m = DeepMLP(width, depth).eval().cuda()
        x = torch.randn(batch, width, device="cuda")
        with torch.no_grad():
            torch.cuda.synchronize()
            m(x)
            torch.cuda.synchronize()
        print(json.dumps({"oom": False, "median_ms": 0.0, "note": "unexpected success (model fit VRAM?)"}))
        return 0
    except torch.cuda.OutOfMemoryError as exc:
        print(json.dumps({"oom": True, "note": str(exc)[:160]}))
        return 0
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"oom": False, "error": f"{type(exc).__name__}: {exc}"}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
