"""Runtime resource provisioning decisions.

Decides how parameters are supplied (resident RAM versus disk streaming) and how
many workers the executor may use. Both decisions are driven by the plan and the
configured budget, never by "use everything" defaults.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from streamcompiler.compile.regions import RegionProgram
from streamcompiler.config import CompileConfig
from streamcompiler.errors import MemoryCapacityError, UnsupportedFeatureError
from streamcompiler.runtime.tensor_store import (
    ParameterStore,
    ResidentParameterStore,
    StreamingParameterStore,
)
from streamcompiler.storage.pack import pack_state_dict

if TYPE_CHECKING:
    from streamcompiler.compile.pipeline import PortableArtifact, SpecializedArtifact


def worker_count(specialized: SpecializedArtifact, config: CompileConfig) -> int:
    """Number of regions that may execute simultaneously.

    Comes from the concurrency measurement taken during specialization, so a plan
    only pays for threads when overlapping regions was observed to be faster.
    """
    if not config.allow_concurrent_regions:
        return 1
    if config.max_concurrent_regions > 0:
        return config.max_concurrent_regions
    decision = specialized.validation.get("concurrency")
    if isinstance(decision, dict):
        return max(1, int(decision.get("workers", 1)))
    return 1


def intraop_threads(specialized: SpecializedArtifact, config: CompileConfig) -> int:
    """Intra-op threads per worker while regions overlap, or 0 to leave it alone.

    Only a measured decision may change the thread count; a forced worker count from
    the config does not imply anything about how the cores should be divided.
    """
    if not config.allow_concurrent_regions:
        return 0
    decision = specialized.validation.get("concurrency")
    if isinstance(decision, dict) and decision.get("enabled"):
        return max(0, int(decision.get("intraop_threads", 0)))
    return 0


def build_parameter_store(
    program: RegionProgram,
    portable: PortableArtifact,
    config: CompileConfig,
    *,
    artifact_dir: Path | None = None,
    pack_lookup_dirs: tuple[Path, ...] = (),
) -> ParameterStore:
    """Choose the cheapest store that satisfies the configured RAM budget."""
    budget = config.ram_budget_bytes
    total = program.total_state_bytes()
    if budget is None or total <= budget:
        return ResidentParameterStore(program.state_tensors())

    if not config.allow_nvme_streaming:
        raise MemoryCapacityError(
            f"Model state is {total} bytes but ram_budget_bytes={budget} and "
            "allow_nvme_streaming=False. Raise the RAM budget or enable disk streaming."
        )

    if config.allow_training:
        # Next slice for larger-than-RAM train: mutable pack writeback after
        # optimizer.step, defer parameter_evict under enable_grad, then region-local
        # backward/recompute so not every weight stays live through full backward.
        raise UnsupportedFeatureError(
            "allow_training=True is incompatible with NVMe parameter streaming "
            f"(model state is {total} bytes, ram_budget_bytes={budget}). "
            "Raise ram_budget_bytes so parameters stay resident, or compile without "
            "allow_training for inference-only streaming."
        )

    required = program.max_region_state_bytes()
    if required > budget:
        raise MemoryCapacityError(
            f"ram_budget_bytes={budget} is smaller than the {required} bytes the largest "
            "region needs resident at once. Lower CompileConfig.max_region_nodes to split "
            "the graph further, or raise the budget."
        )
    pack_path = _ensure_pack(
        program,
        portable,
        config,
        artifact_dir=artifact_dir,
        pack_lookup_dirs=pack_lookup_dirs,
    )
    store = StreamingParameterStore(
        pack_path,
        program.state_bindings,
        budget_bytes=budget,
        # Enable CUDA page-locked staging when accelerators can consume H2D copies.
        pin_memory=bool(torch.cuda.is_available()),
    )
    _release_resident_state(program)
    return store


def _ensure_pack(
    program: RegionProgram,
    portable: PortableArtifact,
    config: CompileConfig,
    *,
    artifact_dir: Path | None = None,
    pack_lookup_dirs: tuple[Path, ...] = (),
) -> Path:
    """Return a model pack path, writing one if the artifact has none."""
    from streamcompiler.errors import StorageError
    from streamcompiler.storage.pack import resolve_pack_path

    for root in pack_lookup_dirs:
        bundled = Path(root) / "model.pack"
        if bundled.exists():
            portable.packed_model_path = str(bundled)
            return bundled.resolve()

    if portable.packed_model_path:
        try:
            return resolve_pack_path(
                portable.packed_model_path,
                artifact_dir=artifact_dir,
                cache_dir=Path(config.cache_dir),
            )
        except StorageError:
            pass
    directory = Path(config.cache_dir) / "packs"
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError:
        directory = Path(tempfile.gettempdir())
    path = directory / f"{portable.name}-{abs(hash(tuple(program.state_bindings)))}.pack"
    pack = pack_state_dict(
        program.state_dict_for_pack(),
        path,
        quantize=bool(config.allow_quantized_storage and config.numerical_mode == "quantized"),
    )
    portable.packed_model_path = str(pack.path)
    return pack.path


def _release_resident_state(program: RegionProgram) -> None:
    """Drop host copies of streamed parameters so the budget is actually enforced.

    Attributes are replaced with zero-element tensors of the same dtype: the
    streamed tensors come from the pack, and any accidental use of the resident
    copy fails loudly on a shape mismatch instead of silently computing garbage.
    """
    import torch

    for env_name, target in program.state_bindings.items():
        parts = target.split(".")
        owner: object = program.root
        for part in parts[:-1]:
            owner = getattr(owner, part)
        leaf = parts[-1]
        current = getattr(owner, leaf, None)
        if not isinstance(current, torch.Tensor):
            continue
        placeholder = torch.empty(0, dtype=current.dtype)
        if isinstance(owner, torch.nn.Module):
            if leaf in owner._parameters:
                owner._parameters[leaf] = torch.nn.Parameter(placeholder, requires_grad=False)
                continue
            if leaf in owner._buffers:
                owner._buffers[leaf] = placeholder
                continue
        setattr(owner, leaf, placeholder)
        del env_name
