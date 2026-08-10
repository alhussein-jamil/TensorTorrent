"""Online profile feedback: fold measured run reports into planner priors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tensortorrent.native import native_available, require_native


@dataclass
class ProfileFeedback:
    """Running averages of measured region latencies from live executions.

    When the native extension is loaded, observations are also stored in
    ``NativeProfileDatabase`` so ``apply_profile_feedback`` can replan from
    durable native records.
    """

    region_latency_s: dict[str, float] = field(default_factory=dict)
    region_device: dict[str, str] = field(default_factory=dict)
    samples: dict[str, int] = field(default_factory=dict)
    transfer_latency_s: list[float] = field(default_factory=list)
    updates: int = 0
    cache_key: str = "live"
    _native_db: Any | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self._native_db is None and native_available():
            self._native_db = require_native().NativeProfileDatabase()

    def observe_report(self, report: Any) -> None:
        events = getattr(report, "events", None) or []
        for event in events:
            rid = getattr(event, "region_id", None)
            if not rid and getattr(event, "opcode", None) == "Compute":
                # Fall back: notes like "Compute region_0" or executable_ref on schedule.
                notes = str(getattr(event, "notes", "") or "")
                if notes.startswith("Compute "):
                    rid = notes.split(" ", 1)[1].split(" ", 1)[0]
            start = getattr(event, "start_s", None)
            end = getattr(event, "end_s", None)
            if rid is None or start is None or end is None:
                continue
            duration = max(0.0, float(end) - float(start))
            n = self.samples.get(rid, 0)
            prev = self.region_latency_s.get(rid, duration)
            self.region_latency_s[rid] = (prev * n + duration) / (n + 1)
            self.samples[rid] = n + 1
            device = getattr(event, "device", None) or getattr(event, "resource", None)
            if device is not None:
                self.region_device[rid] = str(device)
            if self._native_db is not None:
                self._native_db.insert(
                    self.cache_key,
                    str(rid),
                    str(device or "unknown"),
                    float(duration),
                    0,
                    "measured",
                    0.0,
                    0.0,
                )
        store = getattr(report, "parameter_store", None) or {}
        if isinstance(store, dict) and "exposed_io_s" in store:
            self.transfer_latency_s.append(float(store["exposed_io_s"]))
        self.updates += 1

    def prior_for(self, region_id: str, fallback_s: float) -> float:
        if self._native_db is not None:
            native_med = self._native_db.get_region_median(self.cache_key, region_id)
            if native_med is not None:
                return float(native_med)
        return float(self.region_latency_s.get(region_id, fallback_s))

    def merge_into_measurements(self, measurements: Any) -> Any:
        """Override measured latencies for observed (region, device) pairs.

        Returns a new ``MeasurementSet`` so callers can re-plan without mutating
        the compile-time measurements.
        """
        from tensortorrent.compile.measure import MeasurementSet, RegionMeasurement

        out = MeasurementSet()
        for _region_id, by_device in getattr(measurements, "by_region", {}).items():
            for _device, measurement in by_device.items():
                out.add(measurement)
        for region_id, latency in self.region_latency_s.items():
            device = self.region_device.get(region_id)
            if device is None:
                continue
            if self._native_db is not None:
                native_med = self._native_db.get_region_median(self.cache_key, region_id)
                if native_med is not None:
                    latency = float(native_med)
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
        payload = {
            "region_latency_s": dict(self.region_latency_s),
            "region_device": dict(self.region_device),
            "samples": dict(self.samples),
            "transfer_samples": len(self.transfer_latency_s),
            "updates": self.updates,
            "native_profiler": self._native_db is not None,
        }
        if self._native_db is not None:
            try:
                payload["native_stats"] = dict(self._native_db.stats())
            except Exception:  # pragma: no cover
                payload["native_stats"] = {"native_profiler": True}
        return payload
