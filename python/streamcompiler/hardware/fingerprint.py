"""Machine fingerprinting for cache invalidation."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from importlib import metadata
from typing import Any

from streamcompiler.hardware.constants import DEFAULT_SYSTEM_PROBE_TIMEOUT_S


def _safe_run(cmd: list[str]) -> str:
    try:
        out = subprocess.check_output(
            cmd,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=DEFAULT_SYSTEM_PROBE_TIMEOUT_S,
        )
        return out.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def collect_fingerprint_payload() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "cpu_count_logical": os.cpu_count(),
    }
    try:
        import torch

        payload["torch"] = torch.__version__
        payload["torch_cuda"] = torch.version.cuda
        payload["torch_hip"] = getattr(torch.version, "hip", None)
        payload["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            payload["cuda_devices"] = [
                {
                    "name": torch.cuda.get_device_name(i),
                    "capability": torch.cuda.get_device_capability(i),
                    "total_memory": int(torch.cuda.get_device_properties(i).total_memory),
                }
                for i in range(torch.cuda.device_count())
            ]
        xpu = getattr(torch, "xpu", None)
        payload["xpu_available"] = bool(
            xpu is not None and callable(getattr(xpu, "is_available", None)) and xpu.is_available()
        )
        if payload["xpu_available"] and xpu is not None:
            xpu_devices: list[dict[str, Any]] = []
            for index in range(int(xpu.device_count())):
                props = xpu.get_device_properties(index)
                xpu_devices.append(
                    {
                        "name": str(getattr(props, "name", f"xpu:{index}")),
                        "total_memory": int(getattr(props, "total_memory", 0)),
                        "architecture": str(getattr(props, "architecture", getattr(props, "gpu_subslice_count", ""))),
                    }
                )
            payload["xpu_devices"] = xpu_devices
    except Exception as exc:  # noqa: BLE001
        payload["torch_error"] = str(exc)

    payload["nvidia_smi"] = _safe_run(["nvidia-smi", "-L"])
    payload["rocm_smi"] = _safe_run(["rocm-smi", "--showproductname"])
    # Driver / runtime versions affect specialization validity.
    payload["nvidia_driver"] = _safe_run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    payload["intel_gpu_runtime"] = _safe_run(["sycl-ls"]) or _safe_run(["clinfo", "--raw"])

    # Backend plugin identity is part of specialization validity. Record entry
    # point metadata without importing plugin code during fingerprinting.
    try:
        selected = metadata.entry_points().select(group="streamcompiler.backends")
        plugin_rows = [
            {
                "name": ep.name,
                "value": ep.value,
                "distribution": getattr(getattr(ep, "dist", None), "name", ""),
                "version": getattr(getattr(ep, "dist", None), "version", ""),
            }
            for ep in selected
        ]
        payload["backend_plugins"] = sorted(plugin_rows, key=lambda item: (item["name"], item["value"]))
    except Exception as exc:  # noqa: BLE001
        payload["backend_plugin_metadata_error"] = str(exc)
    return payload


def machine_fingerprint(payload: dict[str, Any] | None = None) -> str:
    data = payload if payload is not None else collect_fingerprint_payload()
    blob = json.dumps(data, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:32]
