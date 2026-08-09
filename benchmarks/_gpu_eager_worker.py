"""Child-process GPU eager probe (isolates CUDA OOM from the parent suite)."""

from __future__ import annotations

import json
import sys

import torch

from benchmarks.workloads import DeepMLP


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
        # Feasibility probe only — do not report a timed latency when the model fits.
        print(json.dumps({"oom": False, "fits": True, "note": "fits in VRAM (probe; not timed)"}))
        return 0
    except torch.cuda.OutOfMemoryError as exc:
        print(json.dumps({"oom": True, "fits": False, "note": str(exc)[:160]}))
        return 0
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"oom": False, "error": f"{type(exc).__name__}: {exc}"}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
