"""Isolated worker for VRAM size-sweep cases (one process = one model size)."""

from __future__ import annotations

import gc
import json
import sys
import tempfile
import traceback
from pathlib import Path

import torch
import torch.nn as nn

import streamcompiler as sc
from streamcompiler.errors import PlanningError


class DeepMLP(nn.Module):
    def __init__(self, width: int, layers: int, out_features: int = 8) -> None:
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(width, width) for _ in range(layers))
        self.head = nn.Linear(width, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = torch.relu(layer(x))
        return self.head(x)


def _param_bytes(model: nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in model.parameters())


def _dims_for_target_params(target_bytes: int, *, layers: int) -> int:
    per_layer = max(target_bytes // max(layers, 1), 1)
    width = int((per_layer / 4) ** 0.5)
    return max(64, (width // 64) * 64)


def _cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def main() -> int:
    payload = json.loads(sys.stdin.read())
    mode = payload["mode"]
    vram_bytes = int(payload["vram_bytes"])
    fraction = float(payload["fraction"])
    layers = int(payload["layers"])
    cache = Path(tempfile.mkdtemp(prefix="sc-vram-")) / "cache"

    try:
        if mode == "fit_cuda":
            target = int(vram_bytes * fraction)
            width = _dims_for_target_params(target, layers=layers)
            model = DeepMLP(width, layers).eval()
            params = _param_bytes(model)
            assert params < int(vram_bytes * 0.80), (params, vram_bytes, width, layers)
            x = torch.randn(2, width)
            with torch.no_grad():
                expected = model(x).clone()
            compiled = sc.compile(
                model,
                (x,),
                config=sc.CompileConfig(
                    use_torch_compile=False,
                    measure_regions=False,
                    allow_gpu=True,
                    allow_cpu=True,
                    cache_dir=cache,
                ),
            )
            del model
            _cleanup()
            try:
                devices = set(compiled.specialized.plan.devices_used)
                out = compiled(x)
                err = float((out - expected).abs().max().item())
                torch.testing.assert_close(out, expected, atol=1e-3, rtol=1e-3)
                result = {
                    "ok": True,
                    "on_cuda": any(d.startswith("cuda_gpu_") for d in devices),
                    "devices": sorted(devices),
                    "params_bytes": params,
                    "width": width,
                    "layers": layers,
                    "max_abs_err": err,
                    "cuda_peak_bytes": int(torch.cuda.max_memory_allocated()),
                }
            finally:
                compiled.close()
            print(json.dumps(result))
            return 0

        if mode == "oversize_stream":
            target = int(vram_bytes * fraction)
            width = _dims_for_target_params(target, layers=layers)
            model = DeepMLP(width, layers).eval()
            params = _param_bytes(model)
            assert params > vram_bytes, (params, vram_bytes, width, layers)
            x = torch.randn(2, width)
            with torch.no_grad():
                expected = model(x).clone()
            layer_bytes = width * width * 4 + width * 4
            budget = max(layer_bytes * 2, 32 << 20)
            compiled = sc.compile(
                model,
                (x,),
                config=sc.CompileConfig(
                    use_torch_compile=False,
                    measure_regions=False,
                    allow_gpu=True,
                    allow_cpu=True,
                    ram_budget_bytes=budget,
                    max_region_nodes=1,
                    prefetch_distance=1,
                    cache_dir=cache,
                ),
            )
            del model
            _cleanup()
            try:
                devices = set(compiled.specialized.plan.devices_used)
                store = compiled.executor.parameter_store.stats()
                out = compiled(x)
                err = float((out - expected).abs().max().item())
                torch.testing.assert_close(out, expected, atol=1e-3, rtol=1e-3)
                stats = compiled.last_report.parameter_store
                result = {
                    "ok": True,
                    "on_cuda": any(d.startswith("cuda_gpu_") for d in devices),
                    "on_cpu": any(d.startswith("cpu_") for d in devices),
                    "streaming": store.get("kind") == "streaming",
                    "devices": sorted(devices),
                    "params_bytes": params,
                    "width": width,
                    "layers": layers,
                    "budget_bytes": budget,
                    "peak_resident_bytes": int(stats["peak_resident_bytes"]),
                    "reads": int(stats["reads"]),
                    "max_abs_err": err,
                    "cuda_peak_bytes": int(torch.cuda.max_memory_allocated()),
                }
            finally:
                compiled.close()
            print(json.dumps(result))
            return 0

        if mode == "force_gpu_fail":
            width = _dims_for_target_params(int(vram_bytes * fraction), layers=layers)
            model = DeepMLP(width, layers).eval()
            assert _param_bytes(model) > vram_bytes
            x = torch.randn(2, width)
            raised = None
            try:
                sc.compile(
                    model,
                    (x,),
                    config=sc.CompileConfig(
                        use_torch_compile=False,
                        measure_regions=False,
                        allow_cpu=False,
                        allow_gpu=True,
                        ram_budget_bytes=256 << 20,
                        vram_budget_bytes=256 << 20,
                        max_region_nodes=1,
                        cache_dir=cache,
                    ),
                )
            except PlanningError:
                raised = "PlanningError"
            assert raised == "PlanningError"
            print(json.dumps({"ok": True, "raised": raised, "width": width, "layers": layers}))
            return 0

        raise ValueError(f"unknown mode {mode}")
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
