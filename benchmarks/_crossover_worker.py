"""Child-process worker for one model-size crossover point (bounds parent RSS)."""

from __future__ import annotations

import json
import sys

from benchmarks.suites import measure_one_crossover_point


def main() -> int:
    payload = json.loads(sys.stdin.read())
    result = measure_one_crossover_point(
        width=int(payload["width"]),
        depth=int(payload["depth"]),
        vram_multiple=float(payload["vram_multiple"]),
        vram_bytes=int(payload["vram_bytes"]),
        iters=int(payload.get("iters", 1)),
        warmup=int(payload.get("warmup", 0)),
    )
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
