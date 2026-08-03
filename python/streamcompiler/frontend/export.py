"""PyTorch export frontend and public compile entrypoint."""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import Any

from streamcompiler.compile.pipeline import compile_exported_program
from streamcompiler.config import CompileConfig
from streamcompiler.errors import GraphCaptureError
from streamcompiler.runtime.module import CompiledModule

logger = logging.getLogger(__name__)


def capture_module(model: Any, example_inputs: Any, *, strict: bool = True) -> Any:
    """Capture ``model`` with ``torch.export``.

    Raises :class:`GraphCaptureError` with the underlying reason when the model
    cannot be exported — StreamCompiler never silently falls back to eager.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise GraphCaptureError("PyTorch is required") from exc

    if not isinstance(model, torch.nn.Module):
        raise GraphCaptureError(f"compile() expects a torch.nn.Module, received {type(model).__name__}")
    training_states = tuple((module, bool(module.training)) for module in model.modules())
    model.eval()
    args, kwargs = _split_example_inputs(example_inputs)
    try:
        return torch.export.export(model, args, kwargs, strict=strict)
    except Exception as exc:
        raise GraphCaptureError(
            f"torch.export failed for {type(model).__name__}: {exc}\n"
            "StreamCompiler requires an exportable module; remove data-dependent "
            "control flow or graph breaks."
        ) from exc
    finally:
        # Assign directly: Module.train() recurses and would destroy mixed
        # per-submodule modes while restoring a parent.
        for module, was_training in training_states:
            module.training = was_training


def _split_example_inputs(example_inputs: Any) -> tuple[tuple[Any, ...], dict[str, Any]]:
    if isinstance(example_inputs, tuple):
        if (
            len(example_inputs) == 2
            and isinstance(example_inputs[0], (tuple, list))
            and isinstance(example_inputs[1], dict)
        ):
            return tuple(example_inputs[0]), dict(example_inputs[1])
        return example_inputs, {}
    if isinstance(example_inputs, list):
        return tuple(example_inputs), {}
    return (example_inputs,), {}


def compile(
    model: Any,
    example_inputs: Any,
    *,
    config: CompileConfig | None = None,
    artifact_dir: str | Path | None = None,
    devices: str = "auto",
    machine: Any | None = None,
    measurements: Any | None = None,
    name: str | None = None,
) -> CompiledModule:
    """Compile a PyTorch module into a machine-specialized executable module.

    Capture and specialize run in one call. For large models where the eager
    module must be freed between export and specialization, use
    :func:`capture_module` then :func:`compile_exported` instead.

    When ``artifact_dir`` is set, persists a reloadable bundle (``exported.pt2``,
    ``compile_config.json``, packs) via :meth:`CompiledModule.save`.
    """
    exported = capture_module(model, example_inputs)
    return compile_exported(
        exported,
        config=config,
        artifact_dir=artifact_dir,
        devices=devices,
        machine=machine,
        measurements=measurements,
        name=name or type(model).__name__,
    )


def compile_exported(
    exported: Any,
    *,
    config: CompileConfig | None = None,
    artifact_dir: str | Path | None = None,
    devices: str = "auto",
    machine: Any | None = None,
    measurements: Any | None = None,
    name: str = "model",
) -> CompiledModule:
    """Specialize an already-captured ``ExportedProgram``.

    Pair with :func:`capture_module` when the eager module should be released
    before planning (peak memory). When ``artifact_dir`` is set, persists a
    reloadable bundle via :meth:`CompiledModule.save`.
    """
    config = _apply_device_selection(config or CompileConfig(), devices)
    out_dir = Path(artifact_dir) if artifact_dir else None
    compiled = compile_exported_program(
        exported,
        config=config,
        name=name,
        artifact_dir=out_dir,
        machine=machine,
        measurements=measurements,
    )
    if out_dir is not None:
        compiled.save(out_dir)
    return compiled


def _apply_device_selection(config: CompileConfig, devices: str) -> CompileConfig:
    """Translate ``devices=`` without mutating the caller-owned configuration."""
    cloned = replace(
        config,
        objective_weights=dict(config.objective_weights),
        extra=dict(config.extra),
    )
    selection = (devices or "auto").strip().lower()
    if selection in ("auto", "all", ""):
        return cloned
    if selection == "cpu":
        return replace(cloned, allow_cpu=True, allow_gpu=False, allow_integrated_gpu=False)
    if selection in ("gpu", "cuda", "accelerator"):
        return replace(cloned, allow_cpu=False, allow_gpu=True)
    raise ValueError(f"Unknown devices selection {devices!r}; expected 'auto', 'cpu' or 'gpu'")
