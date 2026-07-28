"""Runtime plan selection based on shapes and resource pressure."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from streamcompiler.planner.maximal import ExecutionPlan
from streamcompiler.planner.plan_family import PlanFamily, select_bucket


@dataclass
class RuntimeContext:
    batch: int
    seq: int
    free_vram_bytes: dict[str, int]
    request_count: int = 1
    available_cpu_cores: int = 1


class PlanSelector:
    def __init__(self, family: PlanFamily) -> None:
        self.family = family

    def select(self, ctx: RuntimeContext) -> Any:
        bucket = select_bucket(ctx.batch, ctx.seq)
        plan = self.family.choose(ctx.batch, ctx.seq)
        if isinstance(plan, ExecutionPlan):
            # Reject plans that need more VRAM than currently free.
            for device, need in plan.predicted_peak_bytes.items():
                free = ctx.free_vram_bytes.get(device)
                if free is not None and need > free:
                    if self.family.fallback and self.family.fallback in self.family.plans:
                        return self.family.plans[self.family.fallback]
                    raise MemoryError(
                        f"Plan for bucket {bucket.name if bucket else '?'} needs {need} on {device}, free={free}"
                    )
        return plan
