"""User-facing compilation and runtime configuration."""

from __future__ import annotations

import logging
import math
import os
import sys
import warnings
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger("tensortorrent.config")


def _default_cache_dir() -> Path:
    """Cache root: ``$TT_CACHE_DIR`` when set, else ``~/.cache/tensortorrent``.

    Read at call time (not import time) so containers and tests can relocate
    the cache without reimporting the package.
    """
    override = os.environ.get("TT_CACHE_DIR", "").strip()
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "tensortorrent"


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
    max_plan_candidates: int = 32
    planner_beam_width: int = 64
    """Maximum non-dominated partial placements retained at each region."""
    planner_candidates_per_device: int = 2
    """Fastest kernel/dtype variants retained per device during joint search."""
    planner_local_search_iters: int = 2
    """Bounded post-search single-region improvement passes (0 disables)."""
    enable_linear_sharding: bool = True
    """Rewrite oversized ``aten.linear`` nodes into exact output-feature shards."""
    max_linear_shards: int = 128
    """Safety cap for automatically generated linear shards per operator."""
    target_inflight_requests: int = 1
    """Expected serving concurrency used when reporting throughput-oriented plans."""
    max_region_nodes: int = 16
    """Longest straight-line chain kept inside a single region."""
    measure_regions: bool = True
    """Benchmark every region on real tensors instead of using priors.

    When ``False``, specialization skips region-input capture and uses cost
    priors only — lower peak memory during compile (important for large models).
    """
    region_measure_iters: int = 3
    allow_concurrent_regions: bool = True
    """Allow independent regions to execute on different workers simultaneously."""
    max_concurrent_regions: int = 0
    """0 means derive the worker count from the devices the planner selected."""
    ram_budget_bytes: int | None = None
    """Host budget for resident parameters. Exceeding it enables disk streaming."""
    vram_budget_bytes: int | None = None
    """Per-device accelerator memory cap. None uses discovered allocatable bytes."""
    activation_budget_bytes: int | None = None
    """Host peak for live activations. Above this, the planner emits schedule
    spill Evict/Load ops (``activation_overflow_policy="spill"`` only)."""
    activation_overflow_policy: str = "spill"  # spill only; recompute rejected
    """Overflow policy. Only ``\"spill\"`` is implemented; ``\"recompute\"`` raises."""
    prefetch_distance: int = 1
    """Minimum configured prefetch distance (>=1 enables double buffering)."""
    adaptive_prefetch: bool = True
    """Increase/decrease prefetch depth from state size, compute time, and RAM budget."""
    storage_io_workers: int = 2
    """Native pack readers used for concurrent positional reads."""
    storage_queue_depth: int = 128
    """Maximum outstanding native prefetch requests before backpressure drops hints."""
    cache_dir: Path = field(default_factory=lambda: _default_cache_dir())
    """Artifact/pack cache root.

    Defaults to ``$TT_CACHE_DIR`` when set, else ``~/.cache/tensortorrent``.
    The environment override matters for read-only container roots and for
    test isolation, where ``$HOME`` is not writable or must not be shared."""
    profile_level: str = "coarse"  # coarse | competitive | full
    """Region kernel selection depth during specialization.

    - ``coarse`` (default): one ``torch.compile`` attempt per region when enabled;
      skips AOTInductor and the interleaved eager/compile/AOT bake-off.
    - ``competitive`` / ``full``: also run AOTInductor (when available) and keep
      the fastest of eager FX, ``torch.compile``, and AOT on example inputs.
    """
    validate_numerics: bool = True
    atol: float = 1e-5
    rtol: float = 1e-5
    use_torch_compile: bool = True
    """Wrap region FX modules with ``torch.compile`` (Inductor) when beneficial.

    On by default under ``profile_level="coarse"``: Inductor is used when it
    compiles; failure falls back to eager FX. Under ``competitive`` / ``full``,
    specialization also measures Inductor vs eager vs AOT and keeps the winner.
    """
    prefer_direct_path: bool = True
    """Use the zero-overhead direct call when the schedule is eligible.

    Eligible: (1) single Compute with resident parameters and no
    Transfer/Load/Evict, or (2) resident multi-region CPU+accelerator dataflow
    with only Compute/Transfer/events/Release. Default on. Set False or
    ``TT_DIRECT_PATH=0`` to force the schedule executor; ``TT_DIRECT_PATH=1``
    forces attempting the direct path.
    """
    torch_compile_backend: str = "inductor"
    """Passed to ``torch.compile(..., backend=...)``. Default is TorchInductor."""
    allow_training: bool = False
    """Opt-in train/eval like a normal ``nn.Module``.

    Default ``False``: module stays on the heterogeneous inference schedule
    (``torch.inference_mode``); ``.train()`` raises. When ``True``: ``.train()``
    runs the same ExecutableSchedule with autograd enabled for ``backward`` /
    optimizer steps; ``.eval()`` returns to the max-performance inference
    schedule with the updated weights. Keeps multi-region partitions (no fused
    single-region collapse). Incompatible with NVMe parameter streaming,
    ``activation_budget_bytes`` spill, and ``process_workers``.
    """
    online_profile_feedback: bool = True
    """Fold measured region latencies from each ``forward`` into running priors."""
    process_workers: int = 0
    """When >0, run concurrent regions in a persistent process pool (Linux fork).

    Linux ``fork`` only — not mixed-vendor process isolation. Off by default;
    thread workers stay the normal path. Requires fork-inherited region callables.
    Incompatible with ``allow_training=True`` (autograd / shared tensors).
    """
    spill_dir: str | None = None
    """Directory for activation spill files. None falls back to TT_SPILL_DIR,
    then <cache_dir>/spill, then a system tempdir."""
    max_total_spill_bytes: int | None = None
    """Maximum bytes the spill subsystem may consume on disk. None = auto."""
    stall_timeout_s: float = 300.0
    """Seconds before a stalled region execution is considered hung. 0 disables."""
    host_memory_reserve_bytes: int | None = None
    """Bytes to reserve from the host memory budget for OS / other processes."""
    vram_headroom_bytes: int | None = None
    """Bytes to reserve per GPU for the display compositor and driver overhead."""
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.objective, Objective):
            try:
                self.objective = Objective(str(self.objective))
            except ValueError as exc:
                raise ValueError(f"Unsupported objective: {self.objective!r}") from exc
        for name in (
            "allow_cpu",
            "allow_gpu",
            "allow_integrated_gpu",
            "allow_mixed_vendor",
            "allow_host_staged_transfers",
            "allow_nvme_streaming",
            "allow_quantized_storage",
            "measure_regions",
            "allow_concurrent_regions",
            "validate_numerics",
            "use_torch_compile",
            "prefer_direct_path",
            "allow_training",
            "online_profile_feedback",
            "adaptive_prefetch",
            "enable_linear_sharding",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool, got {type(getattr(self, name)).__name__}")
        if self.activation_overflow_policy != "spill":
            raise ValueError(
                "activation_overflow_policy must be 'spill'; "
                f"got {self.activation_overflow_policy!r} (recompute is not implemented)"
            )
        if self.numerical_mode not in {"exact", "reduced_precision", "quantized"}:
            raise ValueError(
                "numerical_mode must be one of 'exact', 'reduced_precision', or 'quantized'; "
                f"got {self.numerical_mode!r}"
            )
        if self.profile_level not in {"coarse", "competitive", "full"}:
            raise ValueError(
                f"profile_level must be one of 'coarse', 'competitive', or 'full'; got {self.profile_level!r}"
            )
        if not self.allow_cpu and not self.allow_gpu:
            raise ValueError("At least one of allow_cpu or allow_gpu must be enabled")
        if not self.allow_gpu:
            # Single CPU-only switch: integrated GPUs are a GPU class, so disable
            # them instead of rejecting the common allow_gpu=False call pattern.
            self.allow_integrated_gpu = False

        positive_ints = {
            "max_plan_candidates": self.max_plan_candidates,
            "max_region_nodes": self.max_region_nodes,
            "region_measure_iters": self.region_measure_iters,
            "planner_beam_width": self.planner_beam_width,
            "planner_candidates_per_device": self.planner_candidates_per_device,
            "target_inflight_requests": self.target_inflight_requests,
            "storage_io_workers": self.storage_io_workers,
            "storage_queue_depth": self.storage_queue_depth,
            "max_linear_shards": self.max_linear_shards,
        }
        for name, value in positive_ints.items():
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an int, got {type(value).__name__}")
            if value < 1:
                raise ValueError(f"{name} must be >= 1, got {value!r}")
        if self.enable_linear_sharding and self.max_linear_shards < 2:
            raise ValueError("max_linear_shards must be >= 2 when enable_linear_sharding=True")
        for name in ("max_concurrent_regions", "prefetch_distance", "process_workers", "planner_local_search_iters"):
            count = getattr(self, name)
            if not isinstance(count, int) or isinstance(count, bool):
                raise TypeError(f"{name} must be an int, got {type(count).__name__}")
            if count < 0:
                raise ValueError(f"{name} must be >= 0, got {count!r}")
        for name in ("ram_budget_bytes", "vram_budget_bytes", "activation_budget_bytes"):
            budget = getattr(self, name)
            if budget is not None:
                if not isinstance(budget, int) or isinstance(budget, bool):
                    raise TypeError(f"{name} must be an int or None, got {type(budget).__name__}")
                if budget <= 0:
                    raise ValueError(f"{name} must be > 0 when set, got {budget!r}")
        for name in ("atol", "rtol"):
            tol = float(getattr(self, name))
            if not math.isfinite(tol) or tol < 0:
                raise ValueError(f"{name} must be a finite non-negative number, got {tol!r}")
        if not isinstance(self.objective_weights, dict) or not self.objective_weights:
            raise ValueError("objective_weights must be a non-empty mapping")
        allowed_weights = {"latency", "memory", "throughput"}
        unknown_weights = sorted(set(self.objective_weights) - allowed_weights)
        if unknown_weights:
            raise ValueError(f"objective_weights contains unsupported keys: {unknown_weights}")
        normalized_weights: dict[str, float] = {}
        for name in allowed_weights:
            weight = float(self.objective_weights.get(name, 0.0))
            if not math.isfinite(weight) or weight < 0:
                raise ValueError(f"objective_weights[{name!r}] must be finite and >= 0")
            normalized_weights[name] = weight
        if not any(weight > 0 for weight in normalized_weights.values()):
            raise ValueError("objective_weights must contain at least one positive weight")
        self.objective_weights = normalized_weights
        if not str(self.torch_compile_backend).strip():
            raise ValueError("torch_compile_backend must be non-empty")
        self.cache_dir = Path(self.cache_dir).expanduser()
        if not isinstance(self.extra, dict):
            raise TypeError("extra must be a dict")

        if self.allow_training and int(self.process_workers) > 0:
            from tensortorrent.errors import UnsupportedFeatureError

            raise UnsupportedFeatureError(
                "allow_training=True is incompatible with process_workers>0 "
                "(forked workers detach tensors and break autograd). "
                "Set process_workers=0 for training, or disable allow_training "
                "for max-performance inference."
            )
        if self.allow_training and self.activation_budget_bytes is not None:
            from tensortorrent.errors import UnsupportedFeatureError

            raise UnsupportedFeatureError(
                "allow_training=True is incompatible with activation_budget_bytes "
                "(activation spill/reload replaces tensors and breaks autograd). "
                "Unset activation_budget_bytes for training, or disable allow_training "
                "for inference-only spill."
            )

        # New fields: validate stall_timeout_s
        self.stall_timeout_s = float(self.stall_timeout_s)
        if not math.isfinite(self.stall_timeout_s) or self.stall_timeout_s < 0:
            raise ValueError(f"stall_timeout_s must be a finite non-negative float, got {self.stall_timeout_s!r}")

        # Validate optional int fields
        for name in ("max_total_spill_bytes", "host_memory_reserve_bytes", "vram_headroom_bytes"):
            v = getattr(self, name)
            if v is not None:
                if not isinstance(v, int) or isinstance(v, bool):
                    raise TypeError(f"{name} must be an int or None, got {type(v).__name__}")
                if v <= 0:
                    raise ValueError(f"{name} must be > 0 when set, got {v!r}")

        # process_workers > 0 on non-Linux → ConfigurationError
        if int(self.process_workers) > 0 and sys.platform != "linux":
            from tensortorrent.errors import ConfigurationError

            raise ConfigurationError(
                f"process_workers={self.process_workers} requires Linux fork semantics; "
                f"current platform is {sys.platform!r}. Set process_workers=0 on non-Linux hosts."
            )

        # process_workers > 0 under WSL2 → warn
        if int(self.process_workers) > 0:
            from tensortorrent.hardware.budget import is_wsl2

            if is_wsl2():
                warnings.warn(
                    "process_workers>0 under WSL2: fork() after CUDA initialization is "
                    "unstable and may cause hangs or corruption. "
                    "Set process_workers=0 if you encounter issues.",
                    stacklevel=2,
                )

    @classmethod
    def polite(cls) -> CompileConfig:
        """Preset for shared desktops and machines with display workloads.

        Choices explained:
        - vram_headroom_bytes=1.5 GiB: the display compositor and driver resident
          state typically consume 512 MiB–1 GiB on an active desktop; 1.5 GiB
          gives comfortable headroom without sacrificing too much compute budget.
        - stall_timeout_s=120.0: a 2-minute stall threshold catches hung workloads
          quickly while still allowing moderately slow regions to complete.
        - max_concurrent_regions=1: avoids CPU/GPU contention that would degrade
          interactive responsiveness on a shared machine.
        - prefetch_distance=1: minimal prefetch so streaming only double-buffers;
          reduces peak pinned-memory footprint on memory-constrained desktops.
        """
        _1_5_GiB = int(1.5 * (1 << 30))
        return cls(
            vram_headroom_bytes=_1_5_GiB,
            stall_timeout_s=120.0,
            max_concurrent_regions=1,
            prefetch_distance=1,
        )

    def require_exact_numerics(self) -> bool:
        return self.numerical_mode == "exact"

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize compile knobs for artifact round-trips."""
        return {
            "objective": self.objective.value,
            "objective_weights": dict(self.objective_weights),
            "allow_cpu": self.allow_cpu,
            "allow_gpu": self.allow_gpu,
            "allow_integrated_gpu": self.allow_integrated_gpu,
            "allow_mixed_vendor": self.allow_mixed_vendor,
            "allow_host_staged_transfers": self.allow_host_staged_transfers,
            "allow_nvme_streaming": self.allow_nvme_streaming,
            "allow_quantized_storage": self.allow_quantized_storage,
            "numerical_mode": self.numerical_mode,
            "max_plan_candidates": self.max_plan_candidates,
            "max_region_nodes": self.max_region_nodes,
            "measure_regions": self.measure_regions,
            "region_measure_iters": self.region_measure_iters,
            "planner_beam_width": self.planner_beam_width,
            "planner_candidates_per_device": self.planner_candidates_per_device,
            "planner_local_search_iters": self.planner_local_search_iters,
            "enable_linear_sharding": self.enable_linear_sharding,
            "max_linear_shards": self.max_linear_shards,
            "target_inflight_requests": self.target_inflight_requests,
            "storage_io_workers": self.storage_io_workers,
            "storage_queue_depth": self.storage_queue_depth,
            "allow_concurrent_regions": self.allow_concurrent_regions,
            "max_concurrent_regions": self.max_concurrent_regions,
            "ram_budget_bytes": self.ram_budget_bytes,
            "vram_budget_bytes": self.vram_budget_bytes,
            "activation_budget_bytes": self.activation_budget_bytes,
            "activation_overflow_policy": self.activation_overflow_policy,
            "prefetch_distance": self.prefetch_distance,
            "adaptive_prefetch": self.adaptive_prefetch,
            "cache_dir": str(self.cache_dir),
            "profile_level": self.profile_level,
            "validate_numerics": self.validate_numerics,
            "atol": self.atol,
            "rtol": self.rtol,
            "use_torch_compile": self.use_torch_compile,
            "prefer_direct_path": self.prefer_direct_path,
            "torch_compile_backend": self.torch_compile_backend,
            "allow_training": self.allow_training,
            "online_profile_feedback": self.online_profile_feedback,
            "process_workers": self.process_workers,
            "spill_dir": self.spill_dir,
            "max_total_spill_bytes": self.max_total_spill_bytes,
            "stall_timeout_s": self.stall_timeout_s,
            "host_memory_reserve_bytes": self.host_memory_reserve_bytes,
            "vram_headroom_bytes": self.vram_headroom_bytes,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> CompileConfig:
        """Rebuild a config from :meth:`to_json_dict` output."""
        from dataclasses import fields

        if not isinstance(data, dict):
            raise TypeError(f"CompileConfig JSON must be a dict, got {type(data).__name__}")
        known = {f.name: f for f in fields(cls)}
        unknown = sorted(set(data) - set(known))
        if unknown:
            logger.warning(
                "CompileConfig.from_json_dict: ignoring unknown keys: %s",
                unknown,
            )
        payload: dict[str, Any] = {}
        for key, value in data.items():
            if key not in known:
                continue
            payload[key] = value
        if "objective" in payload and not isinstance(payload["objective"], Objective):
            payload["objective"] = Objective(str(payload["objective"]))
        if "cache_dir" in payload:
            payload["cache_dir"] = Path(payload["cache_dir"])
        for int_key in (
            "max_plan_candidates",
            "max_region_nodes",
            "region_measure_iters",
            "max_concurrent_regions",
            "prefetch_distance",
            "process_workers",
            "planner_beam_width",
            "planner_candidates_per_device",
            "planner_local_search_iters",
            "target_inflight_requests",
            "storage_io_workers",
            "storage_queue_depth",
            "max_linear_shards",
        ):
            if int_key in payload and payload[int_key] is not None:
                value = payload[int_key]
                if not isinstance(value, int) or isinstance(value, bool):
                    raise TypeError(f"{int_key} must be an integer in compile_config.json")
        for bool_key in (
            "allow_cpu",
            "allow_gpu",
            "allow_integrated_gpu",
            "allow_mixed_vendor",
            "allow_host_staged_transfers",
            "allow_nvme_streaming",
            "allow_quantized_storage",
            "measure_regions",
            "allow_concurrent_regions",
            "validate_numerics",
            "use_torch_compile",
            "prefer_direct_path",
            "allow_training",
            "online_profile_feedback",
            "adaptive_prefetch",
            "enable_linear_sharding",
        ):
            if bool_key in payload and not isinstance(payload[bool_key], bool):
                raise TypeError(f"{bool_key} must be a boolean in compile_config.json")
        for opt_int in ("ram_budget_bytes", "vram_budget_bytes", "activation_budget_bytes"):
            if opt_int in payload and payload[opt_int] is not None:
                value = payload[opt_int]
                if not isinstance(value, int) or isinstance(value, bool):
                    raise TypeError(f"{opt_int} must be an integer or null in compile_config.json")
        for float_key in ("atol", "rtol", "stall_timeout_s"):
            if float_key in payload:
                value = payload[float_key]
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    raise TypeError(f"{float_key} must be numeric in compile_config.json")
                payload[float_key] = float(value)
        for opt_int in ("max_total_spill_bytes", "host_memory_reserve_bytes", "vram_headroom_bytes"):
            if opt_int in payload and payload[opt_int] is not None:
                value = payload[opt_int]
                if not isinstance(value, int) or isinstance(value, bool):
                    raise TypeError(f"{opt_int} must be an integer or null in compile_config.json")
        return cls(**payload)
