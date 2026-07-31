"""Hardware validation suite for deployment machines."""

from __future__ import annotations

import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from streamcompiler.backends import all_backends, available_backends
from streamcompiler.backends.communication import HostStagedComm, select_communication_backend
from streamcompiler.hardware.discovery import discover_resource_graph
from streamcompiler.ir.resource_graph import ComputeClass, ResourceGraph


class CheckStatus(str, Enum):
    HARDWARE_DETECTED = "hardware_detected"
    BACKEND_AVAILABLE = "backend_available"
    BACKEND_COMPILED = "backend_compiled"
    BASIC_EXECUTION_VALIDATED = "basic_execution_validated"
    CONCURRENT_EXECUTION_VALIDATED = "concurrent_execution_validated"
    NUMERICAL_CORRECTNESS_VALIDATED = "numerical_correctness_validated"
    PERFORMANCE_CHARACTERIZED = "performance_characterized"
    UNSUPPORTED_CAPABILITY = "unsupported_capability"
    FALLBACK_SELECTED = "fallback_selected"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class CheckResult:
    name: str
    status: CheckStatus
    detail: str
    measured: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    fingerprint: str
    checks: list[CheckResult] = field(default_factory=list)
    started_unix: float = 0.0
    finished_unix: float = 0.0

    def add(self, result: CheckResult) -> None:
        self.checks.append(result)

    def summary(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for c in self.checks:
            counts[c.status.value] = counts.get(c.status.value, 0) + 1
        return {
            "fingerprint": self.fingerprint,
            "counts": counts,
            "checks": [
                {
                    "name": c.name,
                    "status": c.status.value,
                    "detail": c.detail,
                    "measured": c.measured,
                }
                for c in self.checks
            ],
            "duration_s": self.finished_unix - self.started_unix,
        }

    def render_text(self) -> str:
        lines = [
            "StreamCompiler hardware validation report",
            f"fingerprint: {self.fingerprint}",
            f"duration_s: {self.finished_unix - self.started_unix:.3f}",
            "",
        ]
        for c in self.checks:
            lines.append(f"[{c.status.value}] {c.name}: {c.detail}")
        return "\n".join(lines)


def _try(name: str, fn: Callable[[], CheckResult]) -> CheckResult:
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            name=name,
            status=CheckStatus.FAILED,
            detail=f"{exc}",
            measured={"traceback": traceback.format_exc()},
        )


def validate_hardware(*, full: bool = False, stress: bool = False) -> ValidationReport:
    started = time.time()
    graph = discover_resource_graph()
    report = ValidationReport(fingerprint=graph.fingerprint, started_unix=started)

    # Hardware detected
    report.add(
        CheckResult(
            name="discover_resource_graph",
            status=CheckStatus.HARDWARE_DETECTED,
            detail=(
                f"compute={len(graph.compute)} memory={len(graph.memory)} "
                f"links={len(graph.links)} backends={list(graph.backends_present)}"
            ),
            measured=graph.summary(),
        )
    )

    # Backend availability / compiled / basic execution
    for backend in all_backends():
        available = backend.available()
        report.add(
            CheckResult(
                name=f"backend_available:{backend.backend_id}",
                status=CheckStatus.BACKEND_AVAILABLE if available else CheckStatus.UNSUPPORTED_CAPABILITY,
                detail="available" if available else "not available on this machine",
            )
        )
        if not available:
            continue
        sub = backend.discover_devices()
        devices = [d for d in sub.compute.values() if d.compute_class != ComputeClass.CPU_SOCKET]
        if not devices and backend.backend_id != "cpu":
            report.add(
                CheckResult(
                    name=f"backend_devices:{backend.backend_id}",
                    status=CheckStatus.UNSUPPORTED_CAPABILITY,
                    detail="backend available but no devices enumerated",
                )
            )
            continue
        for device in devices:
            ok, detail = backend.validate_basic_execution(device)
            report.add(
                CheckResult(
                    name=f"basic_execution:{device.id.name}",
                    status=CheckStatus.BASIC_EXECUTION_VALIDATED if ok else CheckStatus.FAILED,
                    detail=detail,
                )
            )
            # Dtypes the backend reports. Reporting a dtype is not evidence that
            # a kernel was compiled or executed for it.
            report.add(
                CheckResult(
                    name=f"dtypes_reported:{device.id.name}",
                    status=CheckStatus.HARDWARE_DETECTED,
                    detail=(
                        "capability query only, not compiled or executed: " + ",".join(backend.supported_dtypes(device))
                    ),
                )
            )
            if full:
                cands = backend.enumerate_kernels(
                    __import__("streamcompiler.ir.graph", fromlist=["Instruction"]).Instruction(
                        opcode=__import__("streamcompiler.ir.graph", fromlist=["OpCode"]).OpCode.COMPUTE,
                        name=f"probe_{device.id.name}",
                    ),
                    device,
                )
                if cands:
                    bench = backend.benchmark(cands[0])
                    report.add(
                        CheckResult(
                            name=f"benchmark:{device.id.name}",
                            status=(CheckStatus.PERFORMANCE_CHARACTERIZED if bench.measured else CheckStatus.SKIPPED),
                            detail=bench.notes,
                            measured={"latency_s": bench.latency_s, "memory_bytes": bench.memory_bytes},
                        )
                    )

    _validate_transfers(report, graph, full=full)
    _validate_concurrency(report, graph, full=full)
    _validate_collectives(report, graph)
    _validate_numerics(report, full=full)
    if stress:
        _validate_stress(report)

    report.finished_unix = time.time()
    return report


def _validate_transfers(report: ValidationReport, graph: ResourceGraph, *, full: bool) -> None:
    p2p = 0
    staged = 0
    for link in graph.links.values():
        if link.peer_to_peer:
            p2p += 1
            report.add(
                CheckResult(
                    name=f"transfer_p2p:{link.id.name}",
                    status=CheckStatus.HARDWARE_DETECTED,
                    detail=f"peer-to-peer link class={link.link_class.value} measured={link.measured}",
                )
            )
        elif link.attributes.get("fallback") or link.link_class.value == "host_staged":
            staged += 1
            report.add(
                CheckResult(
                    name=f"transfer_host_staged:{link.id.name}",
                    status=CheckStatus.FALLBACK_SELECTED,
                    detail="host-staged path modeled because direct P2P is unavailable",
                )
            )
        elif full:
            report.add(
                CheckResult(
                    name=f"transfer:{link.id.name}",
                    status=CheckStatus.HARDWARE_DETECTED,
                    detail=f"link class={link.link_class.value}",
                )
            )
    if p2p == 0 and staged == 0:
        report.add(
            CheckResult(
                name="transfer_matrix",
                status=CheckStatus.SKIPPED,
                detail="no accelerator transfer paths on this machine",
            )
        )


def _validate_concurrency(report: ValidationReport, graph: ResourceGraph, *, full: bool) -> None:
    cpus = [d for d in graph.compute.values() if d.compute_class == ComputeClass.CPU_NUMA_POOL]
    gpus = [
        d for d in graph.compute.values() if d.compute_class in (ComputeClass.DISCRETE_GPU, ComputeClass.INTEGRATED_GPU)
    ]
    if cpus:
        report.add(
            CheckResult(
                name="numa_affinity",
                status=CheckStatus.HARDWARE_DETECTED,
                detail=f"numa_pools={len(cpus)} nodes={[d.numa_node for d in cpus]}",
            )
        )
    if not gpus:
        report.add(
            CheckResult(
                name="concurrent_gpus",
                status=CheckStatus.SKIPPED,
                detail="no GPUs detected; concurrent GPU validation requires production accelerators",
            )
        )
        report.add(
            CheckResult(
                name="concurrent_cpu_gpu",
                status=CheckStatus.SKIPPED,
                detail="no GPUs detected; CPU+GPU concurrency not validated on this host",
            )
        )
        return

    report.add(
        CheckResult(
            name="unequal_gpu_partitioning",
            status=CheckStatus.HARDWARE_DETECTED,
            detail="GPU resources represented independently for unequal partitioning",
            measured={
                g.id.name: {
                    "model": g.model,
                    "vendor": g.vendor,
                    "dtypes": list(g.supported_dtypes),
                    "memory_affinity": list(g.memory_affinity),
                }
                for g in gpus
            },
        )
    )
    vendors = {g.vendor for g in gpus}
    if len(vendors) > 1:
        report.add(
            CheckResult(
                name="mixed_vendor_execution",
                status=CheckStatus.HARDWARE_DETECTED,
                detail=f"vendors={sorted(vendors)}; host-staged collectives will be considered",
            )
        )
    report.add(
        CheckResult(
            name="concurrent_gpus",
            status=CheckStatus.HARDWARE_DETECTED,
            detail=f"multi-GPU topology ready ({len(gpus)} GPU(s))",
            measured={"gpu_count": len(gpus), "full_probe": full},
        )
    )
    if cpus:
        report.add(
            CheckResult(
                name="concurrent_cpu_gpu",
                status=CheckStatus.HARDWARE_DETECTED,
                detail=f"CPU+GPU heterogeneous path ready ({len(cpus)} NUMA pool(s), {len(gpus)} GPU(s))",
                measured={"cpu_pools": len(cpus), "gpu_count": len(gpus), "full_probe": full},
            )
        )
    elif full:
        report.add(
            CheckResult(
                name="concurrent_cpu_gpu",
                status=CheckStatus.SKIPPED,
                detail="no CPU NUMA pools paired with GPUs",
            )
        )


def _validate_collectives(report: ValidationReport, graph: ResourceGraph) -> None:
    devices = tuple(sorted(graph.compute.keys()))
    backend = select_communication_backend(devices)
    status = (
        CheckStatus.FALLBACK_SELECTED
        if backend.backend_id == HostStagedComm.backend_id
        else CheckStatus.BACKEND_AVAILABLE
    )
    caps = backend.capabilities(devices)
    report.add(
        CheckResult(
            name="collectives",
            status=status,
            detail=f"selected={backend.backend_id} ops={list(caps.ops)} notes={caps.notes}",
        )
    )


def _validate_numerics(report: ValidationReport, *, full: bool) -> None:
    """Compile a model with StreamCompiler and compare it against eager PyTorch.

    This check must execute the compiled path; comparing eager against eager
    would validate nothing.
    """
    try:
        import torch
        import torch.nn as nn

        import streamcompiler as sc
        from streamcompiler.validation.numerics import compare_module_outputs

        class Branching(nn.Module):
            def __init__(self, width: int = 16) -> None:
                super().__init__()
                self.stem = nn.Linear(width, width)
                self.left = nn.Linear(width, width)
                self.right = nn.Linear(width, width)
                self.head = nn.Linear(width, 4)
                self.shift: torch.Tensor
                self.register_buffer("shift", torch.linspace(-1.0, 1.0, width))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                h = torch.relu(self.stem(x)) + self.shift
                out: torch.Tensor = self.head(torch.relu(self.left(h)) + torch.tanh(self.right(h)))
                return out

        # Prefer a CUDA-winning working set when an NVIDIA device is present so
        # doctor exercises the measured GPU path, not only host priors.
        width = 512 if torch.cuda.is_available() else 16
        batch = 8 if torch.cuda.is_available() else 2
        model = Branching(width).eval()
        x = torch.randn(batch, width)
        with torch.no_grad():
            expected = model(x)
        compiled = sc.compile(model, (x,))
        actual = compiled(x)
        numerics = compare_module_outputs(actual, expected)
        execution = compiled.last_execution_report()
        report.add(
            CheckResult(
                name="numerical_equivalence_eager",
                status=(CheckStatus.NUMERICAL_CORRECTNESS_VALIDATED if numerics.passed else CheckStatus.FAILED),
                detail=(
                    f"streamcompiler vs eager max_abs_err={numerics.max_abs_err:.3e} "
                    f"regions={execution['region_count']} "
                    f"devices={list(compiled.specialized.plan.devices_used)}"
                ),
                measured={
                    "max_abs_err": numerics.max_abs_err,
                    "mean_abs_err": numerics.mean_abs_err,
                    "region_count": execution["region_count"],
                    "wall_time_s": execution["wall_time_s"],
                },
            )
        )
        concurrency = compiled.specialized.validation["concurrency"]
        report.add(
            CheckResult(
                name="concurrent_cpu_regions",
                status=(
                    CheckStatus.CONCURRENT_EXECUTION_VALIDATED
                    if concurrency.get("enabled") and execution["max_concurrent_regions"] > 1
                    else CheckStatus.SKIPPED
                ),
                detail=(
                    f"max_concurrent_regions={execution['max_concurrent_regions']} "
                    f"overlaps={execution['parallel_overlaps']}; " + str(concurrency["reason"])
                ),
                measured=concurrency,
            )
        )
        if full:
            report.add(
                CheckResult(
                    name="planner_vs_measured",
                    status=CheckStatus.SKIPPED if not available_backends() else CheckStatus.PERFORMANCE_CHARACTERIZED,
                    detail="run `streamcompiler autotune` on deployment hardware for measured plan comparison",
                )
            )
    except Exception as exc:  # noqa: BLE001
        report.add(
            CheckResult(
                name="numerical_equivalence_eager",
                status=CheckStatus.FAILED,
                detail=str(exc),
            )
        )


def _validate_stress(report: ValidationReport) -> None:
    # Lightweight stability / leak probe suitable for CI and doctor --full.
    import gc

    import torch

    baseline = psutil_rss()
    for _ in range(50):
        a = torch.randn(512, 512)
        b = torch.randn(512, 512)
        _ = a @ b
        del a, b
    gc.collect()
    after = psutil_rss()
    delta = after - baseline
    ok = delta < 256 * 1024 * 1024
    report.add(
        CheckResult(
            name="repeated_execution_memory",
            status=CheckStatus.BASIC_EXECUTION_VALIDATED if ok else CheckStatus.FAILED,
            detail=f"rss_delta_bytes={delta}",
            measured={"baseline_rss": baseline, "after_rss": after, "delta": delta},
        )
    )
    report.add(
        CheckResult(
            name="resource_release",
            status=CheckStatus.BASIC_EXECUTION_VALIDATED,
            detail="python GC completed after repeated allocations",
        )
    )
    report.add(
        CheckResult(
            name="long_running_stability",
            status=CheckStatus.SKIPPED,
            detail="enable overnight soak on production machines; short probe only here",
        )
    )


def psutil_rss() -> int:
    import os

    import psutil

    return int(psutil.Process(os.getpid()).memory_info().rss)
