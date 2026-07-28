"""Online profile feedback: fold measured run reports into planner priors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProfileFeedback:
    """Running averages of measured region latencies from live executions."""

    region_latency_s: dict[str, float] = field(default_factory=dict)
    region_device: dict[str, str] = field(default_factory=dict)
    samples: dict[str, int] = field(default_factory=dict)
    transfer_latency_s: list[float] = field(default_factory=list)
    updates: int = 0

    def observe_report(self, report: Any) -> None:
        events = getattr(report, "events", None) or []
        for event in events:
            rid = getattr(event, "region_id", None)
            start = getattr(event, "start_s", None)
            end = getattr(event, "end_s", None)
            if rid is None or start is None or end is None:
                continue
            duration = max(0.0, float(end) - float(start))
            n = self.samples.get(rid, 0)
            prev = self.region_latency_s.get(rid, duration)
            self.region_latency_s[rid] = (prev * n + duration) / (n + 1)
            self.samples[rid] = n + 1
            device = getattr(event, "device", None)
            if device is not None:
                self.region_device[rid] = str(device)
        store = getattr(report, "parameter_store", None) or {}
        if isinstance(store, dict) and "exposed_io_s" in store:
            self.transfer_latency_s.append(float(store["exposed_io_s"]))
        self.updates += 1

    def prior_for(self, region_id: str, fallback_s: float) -> float:
        return float(self.region_latency_s.get(region_id, fallback_s))

    def merge_into_measurements(self, measurements: Any) -> Any:
        """Override measured latencies for observed (region, device) pairs.

        Returns a new ``MeasurementSet`` so callers can re-plan without mutating
        the compile-time measurements.
        """
        from streamcompiler.compile.measure import MeasurementSet, RegionMeasurement

        out = MeasurementSet()
        for _region_id, by_device in getattr(measurements, "by_region", {}).items():
            for _device, measurement in by_device.items():
                out.add(measurement)
        for region_id, latency in self.region_latency_s.items():
            device = self.region_device.get(region_id)
            if device is None:
                continue
            out.add(
                RegionMeasurement(
                    region_id=region_id,
                    device=device,
                    backend_id="profile_feedback",
                    latency_s=float(latency),
                    measured=True,
                    notes="online profile feedback prior",
                )
            )
        return out

    def as_dict(self) -> dict[str, Any]:
        return {
            "region_latency_s": dict(self.region_latency_s),
            "region_device": dict(self.region_device),
            "samples": dict(self.samples),
            "transfer_samples": len(self.transfer_latency_s),
            "updates": self.updates,
        }


def refine_contention_from_overlaps(
    *,
    sequential_s: float,
    concurrent_s: float,
    workers: int,
) -> float:
    """Estimate a compute contention multiplier from a measured pair of runs."""
    if sequential_s <= 0 or workers <= 1:
        return 1.0
    ideal = sequential_s / workers
    if ideal <= 0:
        return 1.0
    return max(1.0, float(concurrent_s) / ideal)
