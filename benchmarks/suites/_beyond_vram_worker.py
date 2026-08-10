"""Child-process worker for one forced-GPU beyond-VRAM multiple.

Isolates host RAM retained by earlier multiples so 1.25× / 1.5× are not
false-skipped, and keeps CUDA/export CPU poisoning out of sibling approaches.

When multiple approaches are requested, each runs in its own grandchild
subprocess so back-to-back multi-GiB CPU GEMMs cannot thermally throttle the
next approach (export-free auto otherwise looked 2–3× slower than cpu_eager).
"""

from __future__ import annotations

import json
import sys
from typing import Any


def _timed_run_to_dict(run: Any) -> dict[str, Any]:
    from benchmarks.tooling.harness import TimedRun

    if isinstance(run, TimedRun):
        return {
            "ok": bool(run.ok),
            "median_ms": float(run.median_ms) if run.ok else None,
            "mean_ms": float(run.mean_ms) if run.ok else None,
            "p90_ms": float(getattr(run, "p90_ms", 0.0) or 0.0) if run.ok else None,
            "compile_s": float(run.compile_s) if run.compile_s is not None else None,
            "note": run.note,
            "extras": dict(run.extras or {}),
        }
    if isinstance(run, dict):
        return run
    return {"ok": False, "note": f"unknown run type {type(run).__name__}"}


def _run_one_approach(payload: dict[str, Any], approach: str) -> dict[str, Any]:
    import torch

    import tensortorrent as tt
    from benchmarks.suites.hard_validation import _compile_and_time, _cpu_eager_time
    from benchmarks.suites.memory_hygiene import abort_if_host_tight, load_deepmlp
    from benchmarks.tooling.harness import release_host_memory

    width = int(payload["width"])
    depth = int(payload["depth"])
    mult = float(payload["vram_multiple"])
    vram = int(payload["vram_bytes"])
    wpath = str(payload["weight_path"])
    pbytes = int(payload["params_bytes"])
    batch = int(payload.get("batch", 8))
    iters = int(payload.get("iters", 3))
    warmup = int(payload.get("warmup", 1))

    tight = abort_if_host_tight(pbytes, label=f"forced_gpu_beyond_{mult}_{approach}")
    if tight is not None:
        return {approach: _timed_run_to_dict(tight)}

    x = torch.randn(batch, width)
    ref = load_deepmlp(wpath, width, depth)
    with torch.no_grad():
        expected = ref(x).clone()
    del ref
    release_host_memory()

    m = load_deepmlp(wpath, width, depth)
    try:
        if approach == "cpu_eager":
            run = _cpu_eager_time(m, x, iters=iters, warmup=warmup)
        elif approach == "auto":
            run = _compile_and_time(
                m,
                x,
                config=tt.CompileConfig(
                    use_torch_compile=False,
                    measure_regions=False,
                    allow_gpu=True,
                    allow_cpu=True,
                    vram_budget_bytes=vram,
                    max_region_nodes=16,
                    prefetch_distance=1,
                ),
                label="auto",
                expected=expected,
                iters=iters,
                warmup=warmup,
                instrument=True,
                require_cuda=False,
            )
        elif approach == "forced_gpu":
            run = _compile_and_time(
                m,
                x,
                config=tt.CompileConfig(
                    use_torch_compile=False,
                    measure_regions=False,
                    allow_gpu=True,
                    allow_cpu=False,
                    vram_budget_bytes=vram,
                    max_region_nodes=16,
                    prefetch_distance=1,
                ),
                label="forced_gpu",
                expected=expected,
                iters=iters,
                warmup=warmup,
                instrument=True,
                require_cuda=True,
            )
        else:
            return {approach: {"ok": False, "note": f"unknown approach {approach!r}"}}
    finally:
        del m
        release_host_memory()
        if torch.cuda.is_available():
            with __import__("contextlib").suppress(Exception):
                torch.cuda.empty_cache()

    return {approach: _timed_run_to_dict(run)}


def _run_payload(payload: dict[str, Any]) -> dict[str, Any]:
    from benchmarks.suites.memory_hygiene import abort_if_host_tight, run_json_worker
    from benchmarks.tooling.harness import release_host_memory

    width = int(payload["width"])
    depth = int(payload["depth"])
    mult = float(payload["vram_multiple"])
    vram = int(payload["vram_bytes"])
    pbytes = int(payload["params_bytes"])
    approaches_wanted = list(payload.get("approaches") or ["cpu_eager", "auto", "forced_gpu"])
    # Prefer auto before cpu_eager when both requested: export-free auto is the
    # thermally sensitive path and must not follow a multi-GiB CPU warm-up.
    if "auto" in approaches_wanted and "cpu_eager" in approaches_wanted:
        approaches_wanted = ["auto"] + [a for a in approaches_wanted if a != "auto"]

    tight = abort_if_host_tight(pbytes, label=f"forced_gpu_beyond_{mult}")
    if tight is not None:
        return {
            "vram_multiple": mult,
            "width": width,
            "depth": depth,
            "params_bytes": pbytes,
            "params_over_vram": pbytes / vram,
            "approaches": {"skip": _timed_run_to_dict(tight)},
            "isolated_subprocess": True,
        }

    # Single approach → run in-process (leaf worker).
    if len(approaches_wanted) == 1:
        approaches = _run_one_approach(payload, approaches_wanted[0])
        return {
            "vram_multiple": mult,
            "width": width,
            "depth": depth,
            "params_bytes": pbytes,
            "params_over_vram": pbytes / vram,
            "approaches": approaches,
            "isolated_subprocess": True,
            "isolated_approaches": True,
        }

    # Multiple approaches → one fresh grandchild each (no CUDA bleed).
    # Brief cooldown so multi-GiB CPU GEMM from the prior approach cannot leave
    # the package thermally throttled for the next child.
    import time

    approaches: dict[str, Any] = {}
    for i, approach in enumerate(approaches_wanted):
        if i > 0:
            time.sleep(float(payload.get("approach_cooldown_s") or 8.0))
        release_host_memory()
        child_payload = dict(payload)
        child_payload["approaches"] = [approach]
        code, data, err = run_json_worker(
            "benchmarks.suites._beyond_vram_worker",
            child_payload,
            timeout_s=float(payload.get("approach_timeout_s") or 1800),
        )
        if data is None:
            approaches[approach] = {
                "ok": False,
                "note": (err or f"approach worker failed rc={code}")[-200:],
                "extras": {"mode_label": approach},
            }
        else:
            approaches.update(data.get("approaches") or {})
        release_host_memory()

    return {
        "vram_multiple": mult,
        "width": width,
        "depth": depth,
        "params_bytes": pbytes,
        "params_over_vram": pbytes / vram,
        "approaches": approaches,
        "isolated_subprocess": True,
        "isolated_approaches": True,
    }


def main() -> int:
    payload = json.loads(sys.stdin.read())
    print(json.dumps(_run_payload(payload)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
