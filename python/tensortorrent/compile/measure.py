"""Real region cost measurement.

The planner needs per-region latencies. Instead of inventing constants we run
each region on the tensors it will actually receive, using the example inputs the
user supplied at compile time.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import torch

from tensortorrent.backends.base import KernelCandidate, RegionSource
from tensortorrent.compile.regions import Region, RegionProgram
from tensortorrent.errors import GraphCaptureError


@dataclass
class RegionMeasurement:
    region_id: str
    device: str
    backend_id: str
    latency_s: float
    measured: bool
    notes: str = ""
    simulated: bool = False


@dataclass
class MeasurementSet:
    """Measured region latencies keyed by region and device."""

    by_region: dict[str, dict[str, RegionMeasurement]] = field(default_factory=dict)

    def add(self, measurement: RegionMeasurement) -> None:
        self.by_region.setdefault(measurement.region_id, {})[measurement.device] = measurement

    def get(self, region_id: str, device: str) -> RegionMeasurement | None:
        return self.by_region.get(region_id, {}).get(device)

    def best_usable(self, region_id: str) -> RegionMeasurement | None:
        """Fastest finite measured or simulated probe (never infinite failures)."""
        entries = [
            m
            for m in self.by_region.get(region_id, {}).values()
            if (m.measured or m.simulated) and m.latency_s < float("inf")
        ]
        return min(entries, key=lambda m: m.latency_s) if entries else None

    def as_dict(self) -> dict[str, Any]:
        return {
            rid: {
                dev: {
                    "latency_s": m.latency_s,
                    "measured": m.measured,
                    "simulated": m.simulated,
                    "backend_id": m.backend_id,
                    "notes": m.notes,
                }
                for dev, m in devices.items()
            }
            for rid, devices in self.by_region.items()
        }


def region_source(program: RegionProgram, region: Region, example: tuple[Any, ...] | None = None) -> RegionSource:
    """Build the backend-facing description of one region."""
    dtype = "float32"
    for name in region.outputs:
        spec = program.values.get(name)
        if spec is not None and spec.dtype != "unknown":
            dtype = spec.dtype
            break
    return RegionSource(
        region_id=region.region_id,
        module=program.submodule(region),
        input_names=region.inputs,
        output_names=region.outputs,
        aten_ops=region.aten_ops,
        example_inputs=example,
        attributes={
            "dtype": dtype,
            "node_count": region.node_count,
            "declared_state": tuple(region.state_inputs),
        },
    )


def capture_region_inputs(
    program: RegionProgram,
    flat_inputs: list[Any],
    *,
    time_regions: bool = False,
) -> dict[str, tuple[Any, ...]] | tuple[dict[str, tuple[Any, ...]], dict[str, float]]:
    """Run the program sequentially once and record each region's real inputs.

    This doubles as a correctness check that the lowered dataflow reproduces the
    exported graph: every region input must be produced before it is consumed.

    When ``time_regions=True``, also return wall seconds for the single capture
    forward of each region.
    """
    env: dict[str, Any] = dict(zip(program.user_inputs, flat_inputs, strict=True))
    env.update(program.state_tensors())
    captured: dict[str, tuple[Any, ...]] = {}
    region_times: dict[str, float] = {}
    with torch.inference_mode():
        for region in program.regions:
            args = []
            for name in region.inputs:
                if name not in env:
                    raise GraphCaptureError(
                        f"Region {region.region_id} consumes {name} before it is produced; "
                        "lowering produced an invalid schedule"
                    )
                args.append(env[name])
            captured[region.region_id] = tuple(args)
            start = perf_counter() if time_regions else 0.0
            result = program.submodule(region)(*args)
            if time_regions:
                region_times[region.region_id] = perf_counter() - start
            normalized = _normalize(region.outputs, result)
            if len(normalized) != len(region.outputs):
                raise GraphCaptureError(
                    f"Region {region.region_id} declared {len(region.outputs)} outputs "
                    f"{region.outputs!r} but submodule returned {len(normalized)} values"
                )
            for name, value in zip(region.outputs, normalized, strict=True):
                env[name] = value
    if time_regions:
        return captured, region_times
    return captured


def _normalize(expected: tuple[str, ...], result: Any) -> tuple[Any, ...]:
    """Coerce a submodule result into a tuple matching the declared output names."""
    if not expected:
        return ()
    if isinstance(result, (tuple, list)):
        return tuple(result)
    return (result,)


def profiling_cache_key(
    source: RegionSource,
    candidate: KernelCandidate,
    device: Any,
    *,
    example_inputs: tuple[Any, ...] | None,
) -> str:
    """Stable key for reusing a region measurement.

    Must include the concrete device (not only backend), input shapes, dtype,
    kernel implementation, device fingerprint, and CPU thread configuration so
    results never leak across NUMA nodes, dtypes, or thread pools.
    """
    shapes: list[str] = []
    for value in example_inputs or ():
        if hasattr(value, "shape"):
            shapes.append("x".join(str(int(d)) for d in value.shape))
        else:
            shapes.append(type(value).__name__)
    attrs = getattr(device, "attributes", {}) or {}
    fingerprint = str(attrs.get("fingerprint") or getattr(device, "model", "") or candidate.device)
    threads = attrs.get("intraop_threads")
    if threads is None:
        threads = getattr(device, "concurrency_limit", None) or getattr(device, "core_count", 0)
    return "|".join(
        (
            candidate.region_id,
            candidate.device,
            candidate.backend_id,
            candidate.kernel_id,
            candidate.dtype,
            f"fp={fingerprint}",
            f"threads={int(threads) if threads else 0}",
            f"shapes={','.join(shapes) or 'none'}",
            f"nodes={source.attributes.get('node_count', 0)}",
        )
    )


def _measure_device_regions(
    program: RegionProgram,
    region_inputs: dict[str, tuple[Any, ...]],
    device: Any,
    iters: int,
    cache: dict[str, tuple[float, bool, bool, str]],
) -> list[RegionMeasurement]:
    """Measure every region on one device; thread-local ``cache`` unless merged externally."""
    from tensortorrent.backends import backend_by_id
    from tensortorrent.backends.profiler import profiler_for_backend

    measurements: list[RegionMeasurement] = []
    backend = backend_by_id(device.backend_id)
    if backend is None:
        return measurements
    bench = getattr(backend, "benchmark_region", None)
    profiler = None
    try:
        profiler_kwargs: dict[str, Any] = {}
        if device.backend_id in {"cuda", "rocm", "xpu"}:
            tail = str(device.id.name).rsplit("_", 1)[-1]
            if tail.isdigit():
                profiler_kwargs["device_index"] = int(tail)
        profiler = profiler_for_backend(device.backend_id, **profiler_kwargs)
    except NotImplementedError:
        profiler = None
    if profiler is None and not backend.available():
        return measurements
    if profiler is None and bench is None:
        return measurements

    for region in program.regions:
        example = region_inputs.get(region.region_id)
        if example is None:
            continue
        source = region_source(program, region, example)
        candidate = KernelCandidate(
            region_id=region.region_id,
            device=device.id.name,
            backend_id=device.backend_id,
            kernel_id=f"{device.backend_id}_fx",
            dtype=str(source.attributes.get("dtype", "float32")),
        )
        key = profiling_cache_key(source, candidate, device, example_inputs=example)
        if key in cache:
            latency_s, measured, simulated, notes = cache[key]
            measurements.append(
                RegionMeasurement(
                    region_id=region.region_id,
                    device=device.id.name,
                    backend_id=device.backend_id,
                    latency_s=latency_s,
                    measured=measured,
                    simulated=simulated,
                    notes=f"cache hit; {notes}",
                )
            )
            continue
        try:
            simulated = False
            if profiler is not None:
                module = source.module
                if module is None:
                    raise RuntimeError("region has no module")
                call = module if callable(module) else getattr(module, "forward", module)
                fingerprint = str(device.id.name)
                record = profiler.profile_region(
                    call,
                    example,
                    device_fingerprint=fingerprint,
                    region_graph_hash=key,
                    warm_up=max(1, iters - 1),
                    samples=max(1, iters),
                )
                latency_s = float(record.median_s)
                measured = bool(record.measured)
                simulated = bool(record.simulated)
                notes = (
                    f"backend_profiler:{profiler.backend_id}"
                    f"{':simulated' if simulated else ''}"
                    f" samples={record.sample_count}"
                )
            elif bench is not None:
                result = bench(source, candidate, example, iters=iters)
                latency_s = float(result.latency_s)
                measured = bool(result.measured)
                simulated = bool(getattr(result, "simulated", False))
                notes = result.notes
            else:
                raise RuntimeError(f"no profiler or benchmark_region for backend {device.backend_id}")
        except Exception as exc:  # noqa: BLE001 - a failed probe must not fail compilation
            measurements.append(
                RegionMeasurement(
                    region_id=region.region_id,
                    device=device.id.name,
                    backend_id=device.backend_id,
                    latency_s=float("inf"),
                    measured=False,
                    simulated=False,
                    notes=f"benchmark failed: {exc}",
                )
            )
            continue
        cache[key] = (latency_s, measured, simulated, notes)
        measurements.append(
            RegionMeasurement(
                region_id=region.region_id,
                device=device.id.name,
                backend_id=device.backend_id,
                latency_s=latency_s,
                measured=measured,
                simulated=simulated,
                notes=notes,
            )
        )
    return measurements


def _profileable_devices(devices: list[Any]) -> list[Any]:
    from tensortorrent.backends import backend_by_id
    from tensortorrent.backends.profiler import profiler_for_backend

    out: list[Any] = []
    for device in devices:
        backend = backend_by_id(device.backend_id)
        if backend is None:
            continue
        bench = getattr(backend, "benchmark_region", None)
        profiler = None
        try:
            profiler_kwargs: dict[str, Any] = {}
            if device.backend_id in {"cuda", "rocm", "xpu"}:
                tail = str(device.id.name).rsplit("_", 1)[-1]
                if tail.isdigit():
                    profiler_kwargs["device_index"] = int(tail)
            profiler = profiler_for_backend(device.backend_id, **profiler_kwargs)
        except NotImplementedError:
            profiler = None
        if profiler is None and not backend.available():
            continue
        if profiler is None and bench is None:
            continue
        out.append(device)
    return out


def measure_regions_on_devices(
    program: RegionProgram,
    region_inputs: dict[str, tuple[Any, ...]],
    devices: list[Any],
    *,
    iters: int = 3,
    workers: int = 0,
) -> MeasurementSet:
    """Measure every region on every device whose backend can benchmark regions.

    Prefer :class:`BackendProfiler` for CPU (measured) and mock_accel (simulated);
    fall back to ``backend.benchmark_region`` otherwise.

    CPU devices are always measured serially first so host timings are not
    polluted by concurrent accelerator probe threads (GIL / driver load).
    When ``workers != 1``, distinct accelerator devices are measured in parallel.
    """
    profileable = _profileable_devices(devices)
    results = MeasurementSet()
    cache: dict[str, tuple[float, bool, bool, str]] = {}

    def _is_cpu(device: Any) -> bool:
        return str(getattr(device, "backend_id", "")) in {"cpu", "cpu_numa"}

    cpu_devices = [device for device in profileable if _is_cpu(device)]
    accel_devices = [device for device in profileable if not _is_cpu(device)]

    for device in cpu_devices:
        for measurement in _measure_device_regions(program, region_inputs, device, iters, cache):
            results.add(measurement)

    if not accel_devices:
        return results

    use_parallel = workers != 1 and len(accel_devices) > 1
    if not use_parallel:
        for device in accel_devices:
            for measurement in _measure_device_regions(program, region_inputs, device, iters, cache):
                results.add(measurement)
        return results

    max_workers = workers if workers > 0 else min(len(accel_devices), os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                _measure_device_regions,
                program,
                region_inputs,
                device,
                iters,
                {},
            ): device
            for device in accel_devices
        }
        for future in as_completed(futures):
            for measurement in future.result():
                results.add(measurement)
    return results
