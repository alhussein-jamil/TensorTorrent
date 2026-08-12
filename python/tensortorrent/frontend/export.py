"""PyTorch export frontend and public compile entrypoint."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from tensortorrent.compile.pipeline import compile_exported_program
from tensortorrent.config import CompileConfig
from tensortorrent.errors import GraphCaptureError
from tensortorrent.runtime.module import CompiledModule

logger = logging.getLogger(__name__)


@contextmanager
def _force_pt2_load_device(device: Any) -> Iterator[None]:
    """Force ``torch.export.load`` to materialize tensors on ``device``.

    PT2 has no map_location; archive metadata often says CUDA. We override
    ``deserialize_device`` for the load so weights never touch the accelerator.
    """
    import torch
    from torch.export.pt2_archive import _package as pkg_mod

    target = torch.device(device)
    pkg: Any = pkg_mod
    if not hasattr(pkg, "deserialize_device"):
        raise RuntimeError(
            "torch.export.pt2_archive._package.deserialize_device missing; "
            "cannot force load device — upgrade or pin PyTorch"
        )
    original = pkg.deserialize_device

    def _forced(_meta_device: Any) -> torch.device:
        return target

    pkg.deserialize_device = _forced
    try:
        yield
    finally:
        pkg.deserialize_device = original


def load_exported_program(
    path: str | Path,
    *,
    map_location: Any = "cpu",
) -> Any:
    """Load ``exported.pt2`` with weights on ``map_location`` (default CPU)."""
    import torch

    with _force_pt2_load_device(map_location):
        return torch.export.load(path)


def capture_module(model: Any, example_inputs: Any, *, strict: bool = True) -> Any:
    """Capture ``model`` with ``torch.export``.

    Raises :class:`GraphCaptureError` with the underlying reason when the model
    cannot be exported — TensorTorrent never silently falls back to eager.
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
            "TensorTorrent requires an exportable module; remove data-dependent "
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

    Beyond-VRAM auto mode may skip ``torch.export`` entirely when measured host
    compute already beats predicted parameter streaming — export + CUDA discovery
    otherwise permanently slows multi-GiB CPU matmuls in-process.
    """
    cfg = _apply_device_selection(config or CompileConfig(), devices)
    model_name = name or type(model).__name__
    # Artifact persistence needs a real ExportedProgram; keep the full path.
    if artifact_dir is None and machine is None and measurements is None:
        from tensortorrent.compile.eager_cpu import (
            build_eager_fused_compiled_module,
            should_prefer_eager_cpu_without_export,
        )

        args, kwargs = _split_example_inputs(example_inputs)
        prefer, guard = should_prefer_eager_cpu_without_export(model, args, kwargs, cfg)
        if prefer:
            logger.info(
                "export-free fused CPU baseline for %s (cpu=%.1fms partial_h2d=%.1fms)",
                model_name,
                float(guard.get("cpu_fused_s") or 0.0) * 1e3,
                float(guard.get("streamed_predicted_s") or 0.0) * 1e3,
            )
            return cast(
                CompiledModule,
                build_eager_fused_compiled_module(
                    model,
                    example_inputs,
                    config=cfg,
                    name=model_name,
                    guard=guard,
                ),
            )

    exported = capture_module(model, example_inputs)
    return compile_exported(
        exported,
        config=cfg,
        artifact_dir=artifact_dir,
        devices="auto",  # already applied above
        machine=machine,
        measurements=measurements,
        name=model_name,
        eager_module=model,
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
    eager_module: Any | None = None,
) -> CompiledModule:
    """Specialize an already-captured ``ExportedProgram``.

    Pair with :func:`capture_module` when the eager module should be released
    before planning (peak memory). When ``artifact_dir`` is set, persists a
    reloadable bundle via :meth:`CompiledModule.save`.

    ``eager_module`` keeps the pre-export module for fused-CPU baseline execution
    when auto mode selects host compute over accelerator streaming.
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
        eager_module=eager_module,
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
