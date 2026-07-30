"""PyTorch export frontend and public compile entrypoint."""

from __future__ import annotations

import logging
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
    was_training = bool(model.training)
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
        model.train(was_training)


def _split_example_inputs(example_inputs: Any) -> tuple[tuple[Any, ...], dict[str, Any]]:
    if isinstance(example_inputs, tuple):
        if len(example_inputs) == 2 and isinstance(example_inputs[1], dict):
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
) -> CompiledModule:
    """Compile a PyTorch module into a machine-specialized executable module."""
    config = _apply_device_selection(config or CompileConfig(), devices)
    exported = capture_module(model, example_inputs)
    return compile_exported_program(
        exported,
        config=config,
        name=type(model).__name__,
        artifact_dir=Path(artifact_dir) if artifact_dir else None,
        machine=machine,
        measurements=measurements,
    )


def _apply_device_selection(config: CompileConfig, devices: str) -> CompileConfig:
    """Translate the ``devices=`` shorthand into planner permissions."""
    selection = (devices or "auto").strip().lower()
    if selection in ("auto", "all", ""):
        return config
    if selection == "cpu":
        config.allow_gpu = False
        config.allow_integrated_gpu = False
        return config
    if selection in ("gpu", "cuda", "accelerator"):
        config.allow_cpu = False
        return config
    raise ValueError(f"Unknown devices selection {devices!r}; expected 'auto', 'cpu' or 'gpu'")
