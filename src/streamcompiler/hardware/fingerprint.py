"""Machine fingerprinting for cache invalidation."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from typing import Any


def _safe_run(cmd: list[str]) -> str:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True, timeout=5)
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
    except Exception as exc:  # noqa: BLE001
        payload["torch_error"] = str(exc)

    payload["nvidia_smi"] = _safe_run(["nvidia-smi", "-L"])
    payload["rocm_smi"] = _safe_run(["rocm-smi", "--showproductname"])
    # Driver / runtime versions affect specialization validity.
    payload["nvidia_driver"] = _safe_run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    return payload


def machine_fingerprint(payload: dict[str, Any] | None = None) -> str:
    data = payload if payload is not None else collect_fingerprint_payload()
    blob = json.dumps(data, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:32]
