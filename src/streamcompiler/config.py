"""User-facing compilation and runtime configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class Objective(str, Enum):
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    MEMORY = "memory"
    BALANCED = "balanced"
    WEIGHTED = "weighted"


@dataclass
class CompileConfig:
    """Controls portable compilation and machine specialization."""

    objective: Objective = Objective.LATENCY
    objective_weights: dict[str, float] = field(
        default_factory=lambda: {"latency": 1.0, "memory": 0.0, "throughput": 0.0}
    )
    allow_cpu: bool = True
    allow_gpu: bool = True
    allow_integrated_gpu: bool = True
    allow_mixed_vendor: bool = True
    allow_host_staged_transfers: bool = True
    allow_nvme_streaming: bool = True
    allow_quantized_storage: bool = False
    numerical_mode: str = "exact"  # exact | reduced_precision | quantized
    beam_width: int = 8
    max_plan_candidates: int = 32
    refine_hotspots: bool = True
    max_region_nodes: int = 16
    """Longest straight-line chain kept inside a single region."""
    measure_regions: bool = True
    """Benchmark every region on real tensors instead of using priors."""
    region_measure_iters: int = 3
    allow_concurrent_regions: bool = True
    """Allow independent regions to execute on different workers simultaneously."""
    max_concurrent_regions: int = 0
    """0 means derive the worker count from the devices the planner selected."""
    ram_budget_bytes: int | None = None
    """Host budget for resident parameters. Exceeding it enables disk streaming."""
    prefetch_distance: int = 1
    """How many regions ahead the streaming store prefetches (>=1 double buffers)."""
    cache_dir: Path = field(default_factory=lambda: Path.home() / ".cache" / "streamcompiler")
    profile_level: str = "coarse"  # coarse | competitive | full
    validate_numerics: bool = True
    atol: float = 1e-5
    rtol: float = 1e-5
    extra: dict[str, Any] = field(default_factory=dict)

    def require_exact_numerics(self) -> bool:
        return self.numerical_mode == "exact"
