"""PyTorch export frontend and public compile entrypoint."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from streamcompiler.compile.pipeline import portable_compile_from_ir, specialize_for_machine
from streamcompiler.config import CompileConfig
from streamcompiler.errors import GraphCaptureError
from streamcompiler.frontend.lower import lower_exported_program
from streamcompiler.runtime.module import CompiledModule

logger = logging.getLogger(__name__)


def capture_module(model: Any, example_inputs: Any, *, strict: bool = True) -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise GraphCaptureError("PyTorch is required") from exc

    model.eval()
    try:
        args = example_inputs if isinstance(example_inputs, tuple) else (example_inputs,)
        return torch.export.export(model, args, strict=strict)
    except Exception as exc:
        raise GraphCaptureError(f"torch.export failed: {exc}") from exc


def compile(
    model: Any,
    example_inputs: Any,
    *,
    config: CompileConfig | None = None,
    artifact_dir: str | Path | None = None,
) -> CompiledModule:
    """Full portable compile + machine specialization pipeline."""
    config = config or CompileConfig()
    exported = capture_module(model, example_inputs)
    ir = lower_exported_program(exported, name=type(model).__name__)
    from streamcompiler.frontend.normalize import normalize_graph

    ir = normalize_graph(ir)
    from streamcompiler.analysis import (
        detect_repeated_blocks,
        eliminate_redundancy,
        run_alias_analysis,
        run_liveness_analysis,
    )

    alias = run_alias_analysis(ir)
    live = run_liveness_analysis(ir)
    ir.repeated_blocks = detect_repeated_blocks(ir)
    eliminate_redundancy(ir)
    ir.metadata["alias_groups"] = alias.groups
    ir.metadata["liveness"] = {k: list(v) for k, v in live.intervals.items()}

    state_dict = None
    try:
        state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    except Exception:  # noqa: BLE001
        state_dict = None

    out = Path(artifact_dir) if artifact_dir else None
    portable = portable_compile_from_ir(ir, state_dict=state_dict, output_dir=out)
    specialized = specialize_for_machine(
        portable,
        config=config,
        output_dir=(out / "specialized") if out else None,
    )
    return CompiledModule(portable=portable, specialized=specialized, config=config)
