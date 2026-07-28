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
    cache_dir: Path = field(default_factory=lambda: Path.home() / ".cache" / "streamcompiler")
    profile_level: str = "coarse"  # coarse | competitive | full
    validate_numerics: bool = True
    atol: float = 1e-5
    rtol: float = 1e-5
    extra: dict[str, Any] = field(default_factory=dict)

    def require_exact_numerics(self) -> bool:
        return self.numerical_mode == "exact"
