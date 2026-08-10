"""Parameter-store selection, pack resolution, and streamed-state release."""

from __future__ import annotations

import shutil
import tempfile
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from tensortorrent.compile.fit import needs_parameter_streaming
from tensortorrent.compile.regions import RegionProgram
from tensortorrent.config import CompileConfig
from tensortorrent.errors import (
    DiskSpaceError,
    MemoryCapacityError,
    StorageError,
    UnsupportedFeatureError,
)
from tensortorrent.runtime.pinning import resolve_parameter_pin
from tensortorrent.runtime.tensor_store import (
    ParameterStore,
    ResidentParameterStore,
    StreamingParameterStore,
)
from tensortorrent.storage.pack import pack_state_dict, resolve_pack_path

if TYPE_CHECKING:
    from tensortorrent.compile.pipeline import PortableArtifact


def build_parameter_store(
    program: RegionProgram,
    portable: PortableArtifact,
    config: CompileConfig,
    *,
    artifact_dir: Path | None = None,
    pack_lookup_dirs: tuple[Path, ...] = (),
    pin_memory: bool = False,
    machine: object | None = None,
) -> ParameterStore:
    """Choose the cheapest parameter store that satisfies the RAM budget."""
    budget = config.ram_budget_bytes
    total = program.total_state_bytes()
    streaming = needs_parameter_streaming(config, state_bytes=total)
    use_pin = resolve_parameter_pin(
        wants_pin=bool(pin_memory),
        state_bytes=total,
        machine=machine,
        streaming=streaming,
        allow_training=bool(config.allow_training),
    )

    if not streaming:
        if budget is not None and total > int(budget) and not config.allow_nvme_streaming:
            raise MemoryCapacityError(
                f"Model state is {total} bytes but ram_budget_bytes={budget} and "
                "allow_nvme_streaming=False. Raise the RAM budget or enable disk streaming."
            )
        return ResidentParameterStore(program.state_tensors(), pin_memory=use_pin)

    if config.allow_training:
        raise UnsupportedFeatureError(
            "allow_training=True is incompatible with NVMe parameter streaming "
            f"(model state is {total} bytes, ram_budget_bytes={budget}). "
            "Raise ram_budget_bytes so parameters stay resident, or compile without "
            "allow_training for inference-only streaming."
        )

    if budget is None:
        raise MemoryCapacityError("internal error: NVMe streaming selected without ram_budget_bytes")
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
        pin_memory=use_pin,
        io_workers=config.storage_io_workers,
        queue_limit=config.storage_queue_depth,
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
    """Return a model pack path, creating one only when none can be reused."""
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
        except StorageError as exc:
            warnings.warn(
                f"resolve_pack_path raised StorageError: {exc}. Falling back to writing a new pack file.",
                stacklevel=4,
            )

    directory = _pack_directory(config)
    path = directory / f"{portable.name}-{abs(hash(tuple(program.state_bindings)))}.pack"
    _check_pack_disk_space(directory, program.total_state_bytes() or 1)
    pack = pack_state_dict(
        program.state_dict_for_pack(),
        path,
        quantize=bool(config.allow_quantized_storage and config.numerical_mode == "quantized"),
    )
    portable.packed_model_path = str(pack.path)
    return pack.path


def _pack_directory(config: CompileConfig) -> Path:
    directory = Path(config.cache_dir) / "packs"
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError:
        directory = Path(tempfile.gettempdir())
    return directory


def _check_pack_disk_space(directory: Path, needed: int) -> None:
    try:
        free = int(shutil.disk_usage(str(directory)).free)
    except OSError:
        return
    if needed > free:
        raise DiskSpaceError(directory, needed, free)


def _release_resident_state(program: RegionProgram) -> None:
    """Replace streamed parameters and buffers with zero-sized sentinels."""
    for target in program.state_bindings.values():
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
