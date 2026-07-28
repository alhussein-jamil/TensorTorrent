"""Runtime resource provisioning decisions.

Decides how parameters are supplied (resident RAM versus disk streaming) and how
many workers the executor may use. Both decisions are driven by the plan and the
configured budget, never by "use everything" defaults.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from streamcompiler.codegen.regions import RegionProgram
from streamcompiler.config import CompileConfig
from streamcompiler.planner.maximal import ExecutionPlan
from streamcompiler.runtime.tensor_store import (
    ParameterStore,
    ResidentParameterStore,
    StreamingParameterStore,
)
from streamcompiler.storage.pack import pack_state_dict

if TYPE_CHECKING:
    from streamcompiler.compile.pipeline import PortableArtifact


def worker_count(plan: ExecutionPlan, config: CompileConfig) -> int:
    """Number of regions that may execute simultaneously.

    Capped by the devices the planner actually selected: a single-device plan
    keeps the sequential fast path, avoiding thread overhead that would make
    execution slower rather than faster.
    """
    if not config.allow_concurrent_regions:
        return 1
    if config.max_concurrent_regions > 0:
        return config.max_concurrent_regions
    devices = len(plan.devices_used) or 1
    return max(1, devices)


def build_parameter_store(
    program: RegionProgram,
    portable: PortableArtifact,
    config: CompileConfig,
) -> ParameterStore:
    """Choose the cheapest store that satisfies the configured RAM budget."""
    budget = config.ram_budget_bytes
    total = program.total_state_bytes()
    if budget is None or total <= budget:
        return ResidentParameterStore(program.state_tensors())

    pack_path = _ensure_pack(program, portable, config)
    store = StreamingParameterStore(
        pack_path,
        program.state_bindings,
        budget_bytes=budget,
        pin_memory=False,
    )
    _release_resident_state(program)
    return store


def _ensure_pack(program: RegionProgram, portable: PortableArtifact, config: CompileConfig) -> Path:
    """Return a model pack path, writing one if the artifact has none."""
    if portable.packed_model_path:
        candidate = Path(portable.packed_model_path)
        if candidate.exists():
            return candidate
    directory = Path(config.cache_dir) / "packs"
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError:
        directory = Path(tempfile.gettempdir())
    path = directory / f"{portable.name}-{abs(hash(tuple(program.state_bindings)))}.pack"
    pack = pack_state_dict(
        {target: program.state_tensor(env) for env, target in program.state_bindings.items()},
        path,
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
