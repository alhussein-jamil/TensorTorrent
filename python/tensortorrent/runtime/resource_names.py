"""Conservative classification for built-in runtime resource identifiers."""

from __future__ import annotations


def is_device_resource(resource: str) -> bool:
    """Return whether a built-in resource denotes accelerator memory/compute."""
    name = resource.strip().lower()
    return name.startswith(("cuda", "rocm", "xpu", "gpu", "vram", "mock")) or any(
        marker in name for marker in ("_cuda_", "_rocm_", "_xpu_", "_gpu_", "_vram_", "_mock_")
    )


def is_host_resource(resource: str) -> bool:
    """Return whether a built-in resource denotes CPU, RAM, or storage."""
    name = resource.strip().lower()
    return name in {"cpu", "host", "ram", "disk", "nvme", "storage"} or name.startswith(
        (
            "cpu_",
            "cpu:",
            "numa_",
            "numa:",
            "host_",
            "host:",
            "system_ram",
            "ram_",
            "pinned_",
            "pageable_",
            "disk_",
            "nvme_",
            "storage_",
        )
    )
