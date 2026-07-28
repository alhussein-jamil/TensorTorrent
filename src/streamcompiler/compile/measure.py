"""Real region cost measurement.

The planner needs per-region latencies. Instead of inventing constants we run
each region on the tensors it will actually receive, using the example inputs the
user supplied at compile time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from streamcompiler.backends.base import KernelCandidate, RegionSource
from streamcompiler.codegen.regions import Region, RegionProgram
from streamcompiler.errors import GraphCaptureError


@dataclass
class RegionMeasurement:
    region_id: str
    device: str
    backend_id: str
    latency_s: float
    measured: bool
    notes: str = ""


@dataclass
class MeasurementSet:
    """Measured region latencies keyed by region and device."""

    by_region: dict[str, dict[str, RegionMeasurement]] = field(default_factory=dict)

    def add(self, measurement: RegionMeasurement) -> None:
        self.by_region.setdefault(measurement.region_id, {})[measurement.device] = measurement

    def get(self, region_id: str, device: str) -> RegionMeasurement | None:
        return self.by_region.get(region_id, {}).get(device)

    def best_measured(self, region_id: str) -> RegionMeasurement | None:
        entries = [m for m in self.by_region.get(region_id, {}).values() if m.measured]
        return min(entries, key=lambda m: m.latency_s) if entries else None

    def measured_region_ids(self) -> tuple[str, ...]:
        return tuple(rid for rid in self.by_region if self.best_measured(rid) is not None)

    def as_dict(self) -> dict[str, Any]:
        return {
            rid: {
                dev: {
                    "latency_s": m.latency_s,
                    "measured": m.measured,
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
        attributes={"dtype": dtype, "node_count": region.node_count},
    )


def capture_region_inputs(program: RegionProgram, flat_inputs: list[Any]) -> dict[str, tuple[Any, ...]]:
    """Run the program sequentially once and record each region's real inputs.

    This doubles as a correctness check that the lowered dataflow reproduces the
    exported graph: every region input must be produced before it is consumed.
    """
    env: dict[str, Any] = dict(zip(program.user_inputs, flat_inputs, strict=True))
    env.update(program.state_tensors())
    captured: dict[str, tuple[Any, ...]] = {}
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
            result = program.submodule(region)(*args)
            for name, value in zip(region.outputs, _normalize(region.outputs, result), strict=True):
                env[name] = value
    return captured


def reference_outputs(program: RegionProgram, flat_inputs: list[Any]) -> list[Any]:
    """Compute the program's flat outputs with a plain sequential interpreter."""
    env: dict[str, Any] = dict(zip(program.user_inputs, flat_inputs, strict=True))
    env.update(program.state_tensors())
    with torch.inference_mode():
        for region in program.regions:
            args = tuple(env[name] for name in region.inputs)
            result = program.submodule(region)(*args)
            for name, value in zip(region.outputs, _normalize(region.outputs, result), strict=True):
                env[name] = value
    flat: list[Any] = []
    for kind, ref in program.output_refs:
        flat.append(ref if kind == "constant" else env[str(ref)])
    return flat


def _normalize(expected: tuple[str, ...], result: Any) -> tuple[Any, ...]:
    """Coerce a submodule result into a tuple matching the declared output names."""
    if not expected:
        return ()
    if isinstance(result, (tuple, list)):
        return tuple(result)
    return (result,)


def measure_regions_on_devices(
    program: RegionProgram,
    region_inputs: dict[str, tuple[Any, ...]],
    devices: list[Any],
    *,
    iters: int = 3,
) -> MeasurementSet:
    """Measure every region on every device whose backend can benchmark regions."""
    from streamcompiler.backends import backend_by_id

    results = MeasurementSet()
    cache: dict[tuple[str, str], float] = {}
    for device in devices:
        backend = backend_by_id(device.backend_id)
        if backend is None or not backend.available():
            continue
        bench = getattr(backend, "benchmark_region", None)
        if bench is None:
            continue
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
            key = (region.region_id, device.backend_id)
            if key in cache:
                results.add(
                    RegionMeasurement(
                        region_id=region.region_id,
                        device=device.id.name,
                        backend_id=device.backend_id,
                        latency_s=cache[key],
                        measured=True,
                        notes=f"measured on an identical {device.backend_id} device",
                    )
                )
                continue
            try:
                result = bench(source, candidate, example, iters=iters)
            except Exception as exc:  # noqa: BLE001 - a failed probe must not fail compilation
                results.add(
                    RegionMeasurement(
                        region_id=region.region_id,
                        device=device.id.name,
                        backend_id=device.backend_id,
                        latency_s=float("inf"),
                        measured=False,
                        notes=f"benchmark failed: {exc}",
                    )
                )
                continue
            cache[key] = result.latency_s
            results.add(
                RegionMeasurement(
                    region_id=region.region_id,
                    device=device.id.name,
                    backend_id=device.backend_id,
                    latency_s=result.latency_s,
                    measured=result.measured,
                    notes=result.notes,
                )
            )
    return results
