"""Host-side cost priors for the simulator and VirtualBackend.

Measurements are CPU/host only — never labelled as CUDA or device peak bandwidth.
"""

from __future__ import annotations

import contextlib
import time
from typing import Any

import torch

from tensortorrent.planner.cost.transfer import TransferModel, measure_host_copy


def prediction_error(wall_s: float, predicted_s: float | None) -> dict[str, float | None]:
    """Absolute and relative error: wall − predicted (None when no prediction)."""
    if predicted_s is None:
        return {"prediction_error_s": None, "prediction_relative_error": None}
    err = float(wall_s) - float(predicted_s)
    rel = None if float(predicted_s) <= 0.0 else err / float(predicted_s)
    return {"prediction_error_s": err, "prediction_relative_error": rel}


def runtime_predicted_makespan_s(analytic_makespan_s: float, *, n_compute: int) -> float:
    """Analytic DES makespan plus measured host-bridge tax for runtime prediction."""
    priors = calibrate_host_priors()
    bridge_fixed = float(priors.get("bridge_fixed_s") or 0.0)
    base_compute = max(1, int(priors.get("bridge_base_compute") or 1))
    host_tax = bridge_fixed * max(1.0, float(max(0, n_compute)) / float(base_compute))
    return float(analytic_makespan_s) + host_tax


_HOST_PRIOR_CACHE: dict[str, Any] | None = None


def cached_host_priors() -> dict[str, Any]:
    """Return a copy of the last successful calibration, or {} if none."""
    return dict(_HOST_PRIOR_CACHE) if _HOST_PRIOR_CACHE is not None else {}


def calibrate_host_priors(
    *,
    source: str = "host",
    destination: str = "host",
    sizes: tuple[int, ...] = (1 << 20, 4 << 20, 16 << 20),
    force: bool = False,
) -> dict[str, Any]:
    """Measure host copy alpha/beta plus cheap CPU / GIL noop sample timings.

    Returns a dict suitable for CLI/bench JSON and for seeding VirtualBackend
    transfer priors when topology links lack measured coefficients. Cached
    process-wide after the first call (pass ``force=True`` to remeasure).
    """
    global _HOST_PRIOR_CACHE
    if _HOST_PRIOR_CACHE is not None and not force:
        return dict(_HOST_PRIOR_CACHE)

    # Placeholder prevents recursion when bridge measurement compiles a module
    # (planner priors call host_cpu_region_prior_s → calibrate_host_priors).
    # Cleared on failure so a broken measure never sticks as invent priors.
    _HOST_PRIOR_CACHE = {
        "source": source,
        "destination": destination,
        "alpha_s": 0.0,
        "beta_bytes_per_s": 4e9,
        "measured": False,
        "host_copy_samples": [],
        "cpu_region_s": 5e-5,
        "gil_noop_s": 5e-8,
        "callback_s": 2e-7,
        "bridge_fixed_s": 5e-4,
        "bridge_base_compute": 1,
    }
    try:
        model: TransferModel = measure_host_copy(source, destination, sizes=sizes)

        # Representative host Linear (matches streaming microbench region shape).
        lin = torch.nn.Linear(64, 64).eval()
        x = torch.randn(16, 64)
        with torch.inference_mode():
            for _ in range(5):
                lin(x)
            t0 = time.perf_counter()
            iters = 40
            for _ in range(iters):
                lin(x)
            cpu_region_s = (time.perf_counter() - t0) / iters

        # GIL noop: empty Python call under the interpreter lock.
        def _noop() -> None:
            return None

        for _ in range(50):
            _noop()
        t0 = time.perf_counter()
        gil_iters = 1000
        for _ in range(gil_iters):
            _noop()
        gil_noop_s = (time.perf_counter() - t0) / gil_iters

        # Native schedule tax: one PyO3 callback round-trip approximation.
        def _callback_like(batch: list[str]) -> list[int]:
            return [0 for _ in batch]

        batch = ["t0", "t1", "t2", "t3"]
        for _ in range(50):
            _callback_like(batch)
        t0 = time.perf_counter()
        cb_iters = 2000
        for _ in range(cb_iters):
            _callback_like(batch)
        callback_s = (time.perf_counter() - t0) / cb_iters

        # Fixed host-bridge tax from a one-op resident native forward (not per-op).
        bridge_fixed_s, bridge_base_compute = _measure_native_bridge_fixed_s(cpu_region_s)

        out = {
            "source": source,
            "destination": destination,
            "alpha_s": float(model.alpha_s),
            "beta_bytes_per_s": None if model.beta_bytes_per_s is None else float(model.beta_bytes_per_s),
            "measured": bool(model.measured),
            "host_copy_samples": [
                {"nbytes": s.nbytes, "latency_s": s.latency_s, "notes": s.notes} for s in model.samples
            ],
            "cpu_region_s": float(cpu_region_s),
            "gil_noop_s": float(gil_noop_s),
            "callback_s": float(callback_s),
            "bridge_fixed_s": float(bridge_fixed_s),
            "bridge_base_compute": int(bridge_base_compute),
        }
        _HOST_PRIOR_CACHE = dict(out)
        return dict(out)
    except Exception:
        _HOST_PRIOR_CACHE = None
        raise


def host_cpu_region_prior_s() -> float:
    """Absolute seconds for an unmeasured CPU region prior (measured host Linear)."""
    return float(calibrate_host_priors()["cpu_region_s"])


def _measure_native_bridge_fixed_s(cpu_region_s: float) -> tuple[float, int]:
    """Measure fixed native bridge tax + Compute count from a one-op Linear."""
    fallback = (max(1e-6, float(cpu_region_s) * 40.0), 1)
    try:
        import tensortorrent as tt
        from tensortorrent.config import CompileConfig
        from tensortorrent.ir.graph import OpCode
        from tensortorrent.native import native_available
    except Exception:  # noqa: BLE001
        return fallback
    if not native_available():
        return fallback
    compiled = None
    try:
        model = torch.nn.Linear(64, 64).eval()
        x = torch.randn(8, 64)
        compiled = tt.compile(
            model,
            (x,),
            config=CompileConfig(use_torch_compile=False, measure_regions=False),
        )
        with torch.inference_mode():
            for _ in range(3):
                compiled(x)
            t0 = time.perf_counter()
            iters = 20
            for _ in range(iters):
                compiled(x)
            wall = (time.perf_counter() - t0) / iters
        se = compiled.executor._schedule_executor
        if se is None:
            return fallback
        n_compute = 0
        for inst in se.schedule.instructions:
            if inst.opcode == OpCode.COMPUTE:
                n_compute += 1
        tax = max(0.0, wall - float(cpu_region_s))
        return max(1e-6, tax), max(1, n_compute)
    except Exception:  # noqa: BLE001
        return fallback
    finally:
        if compiled is not None:
            with contextlib.suppress(Exception):
                compiled.close()
