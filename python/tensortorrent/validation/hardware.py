"""Hardware validation suite for deployment machines."""

from __future__ import annotations

import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from tensortorrent.backends import all_backends, available_backends, plugin_errors
from tensortorrent.backends.communication import HostStagedComm, select_communication_backend
from tensortorrent.hardware.discovery import discover_resource_graph
from tensortorrent.ir.resource_graph import ComputeClass, ResourceGraph


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
        ready, blockers = self.production_ready()
        return {
            "fingerprint": self.fingerprint,
            "counts": counts,
            "production_ready": ready,
            "production_blockers": blockers,
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
            "budgets": _resolve_budgets_for_report(),
        }

    def production_ready(self) -> tuple[bool, list[str]]:
        """Return whether this host is ready for production inference.

        ``hardware_detected`` alone is never enough. Every available accelerator
        backend must have measured basic execution, and numerical correctness
        against eager must pass. Unsupported/skipped backends are not blockers.
        """
        blockers: list[str] = []
        failed = [c for c in self.checks if c.status is CheckStatus.FAILED]
        for c in failed:
            blockers.append(f"failed:{c.name}: {c.detail}")

        available_accel: set[str] = set()
        for c in self.checks:
            if not c.name.startswith("backend_available:"):
                continue
            if c.status is not CheckStatus.BACKEND_AVAILABLE:
                continue
            backend_id = c.name.split(":", 1)[1]
            if backend_id in {"cpu", "mock_accel"}:
                continue
            available_accel.add(backend_id)

        for backend_id in sorted(available_accel):
            prefix = {
                "cuda": "basic_execution:cuda_gpu_",
                "rocm": "basic_execution:rocm_gpu_",
                "xpu": "basic_execution:xpu_",
            }.get(backend_id)
            if prefix is None:
                # Plugin / unknown accelerator: any basic_execution with that backend id.
                measured = [
                    c
                    for c in self.checks
                    if c.name.startswith("basic_execution:")
                    and backend_id in c.name
                    and c.status is CheckStatus.BASIC_EXECUTION_VALIDATED
                ]
            else:
                measured = [
                    c
                    for c in self.checks
                    if c.name.startswith(prefix) and c.status is CheckStatus.BASIC_EXECUTION_VALIDATED
                ]
            if not measured:
                blockers.append(
                    f"backend {backend_id!r} available but no measured basic_execution "
                    f"(discovery alone is not production-ready)"
                )

        numerics = next((c for c in self.checks if c.name == "numerical_equivalence_eager"), None)
        if numerics is None or numerics.status is not CheckStatus.NUMERICAL_CORRECTNESS_VALIDATED:
            blockers.append("numerical_equivalence_eager not validated")

        return (not blockers, blockers)

    def render_text(self) -> str:
        lines = [
            "TensorTorrent hardware validation report",
            f"fingerprint: {self.fingerprint}",
            f"duration_s: {self.finished_unix - self.started_unix:.3f}",
            "",
        ]
        for c in self.checks:
            lines.append(f"[{c.status.value}] {c.name}: {c.detail}")
        ready, blockers = self.production_ready()
        lines.append("")
        if ready:
            lines.append("[production_ready] yes — measured execution + numerics for enabled backends")
        else:
            lines.append("[production_ready] no")
            for reason in blockers:
                lines.append(f"  - {reason}")
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


def validate_hardware(*, full: bool = False, stress: bool = False, overnight: bool = False) -> ValidationReport:
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

    # Backend availability / compiled / basic execution. Optional and third-party
    # backends are isolation boundaries: one broken plugin must not prevent the
    # rest of the target machine from being diagnosed.
    for backend in all_backends():
        try:
            available = bool(backend.available())
        except Exception as exc:  # noqa: BLE001
            report.add(
                CheckResult(
                    name=f"backend_available:{backend.backend_id}",
                    status=CheckStatus.FAILED,
                    detail=f"availability probe failed: {type(exc).__name__}: {exc}",
                )
            )
            continue
        report.add(
            CheckResult(
                name=f"backend_available:{backend.backend_id}",
                status=CheckStatus.BACKEND_AVAILABLE if available else CheckStatus.UNSUPPORTED_CAPABILITY,
                detail="available" if available else "not available on this machine",
            )
        )
        if not available:
            continue
        try:
            sub = backend.discover_devices()
        except Exception as exc:  # noqa: BLE001
            report.add(
                CheckResult(
                    name=f"backend_devices:{backend.backend_id}",
                    status=CheckStatus.FAILED,
                    detail=f"device discovery failed: {type(exc).__name__}: {exc}",
                )
            )
            continue
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
            try:
                ok, detail = backend.validate_basic_execution(device)
            except Exception as exc:  # noqa: BLE001
                ok, detail = False, f"validation raised {type(exc).__name__}: {exc}"
            report.add(
                CheckResult(
                    name=f"basic_execution:{device.id.name}",
                    status=CheckStatus.BASIC_EXECUTION_VALIDATED if ok else CheckStatus.FAILED,
                    detail=detail,
                )
            )
            try:
                reported_dtypes = backend.supported_dtypes(device)
            except Exception as exc:  # noqa: BLE001
                reported_dtypes = ()
                report.add(
                    CheckResult(
                        name=f"dtypes_reported:{device.id.name}",
                        status=CheckStatus.FAILED,
                        detail=f"dtype capability query failed: {type(exc).__name__}: {exc}",
                    )
                )
            else:
                report.add(
                    CheckResult(
                        name=f"dtypes_reported:{device.id.name}",
                        status=CheckStatus.HARDWARE_DETECTED,
                        detail="capability query only, not compiled or executed: " + ",".join(reported_dtypes),
                    )
                )
            if full:
                try:
                    cands = backend.enumerate_kernels(
                        __import__("tensortorrent.ir.graph", fromlist=["Instruction"]).Instruction(
                            opcode=__import__("tensortorrent.ir.graph", fromlist=["OpCode"]).OpCode.COMPUTE,
                            name=f"probe_{device.id.name}",
                        ),
                        device,
                    )
                    if cands:
                        bench = backend.benchmark(cands[0])
                        report.add(
                            CheckResult(
                                name=f"benchmark:{device.id.name}",
                                status=(
                                    CheckStatus.PERFORMANCE_CHARACTERIZED if bench.measured else CheckStatus.SKIPPED
                                ),
                                detail=bench.notes,
                                measured={"latency_s": bench.latency_s, "memory_bytes": bench.memory_bytes},
                            )
                        )
                except Exception as exc:  # noqa: BLE001
                    report.add(
                        CheckResult(
                            name=f"benchmark:{device.id.name}",
                            status=CheckStatus.FAILED,
                            detail=f"benchmark probe failed: {type(exc).__name__}: {exc}",
                        )
                    )

    for label, detail in sorted(plugin_errors().items()):
        report.add(
            CheckResult(
                name=f"backend_plugin:{label}",
                status=CheckStatus.FAILED,
                detail=detail,
            )
        )

    _validate_transfers(report, graph, full=full)
    _validate_concurrency(report, graph, full=full)
    _validate_collectives(report, graph)
    _validate_numerics(report, full=full)
    if stress or overnight:
        _validate_stress(report, overnight=overnight)

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
        d
        for d in graph.compute.values()
        if d.compute_class
        in (
            ComputeClass.DISCRETE_GPU,
            ComputeClass.INTEGRATED_GPU,
            ComputeClass.ACCELERATOR,
        )
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
    vendors = {g.vendor or g.backend_id for g in gpus}
    if len(vendors) > 1:
        report.add(
            CheckResult(
                name="mixed_vendor_execution",
                status=CheckStatus.SKIPPED if not full else CheckStatus.HARDWARE_DETECTED,
                detail=(
                    f"vendors={sorted(vendors)}; topology observed only — "
                    f"live mixed-vendor execution requires silicon from each vendor"
                    if not full
                    else f"vendors={sorted(vendors)}; host-staged collectives considered (not a measured pass)"
                ),
            )
        )
    if len(gpus) >= 2:
        if full:
            report.add(_try("concurrent_gpus", _probe_multi_gpu_measured))
        else:
            report.add(
                CheckResult(
                    name="concurrent_gpus",
                    status=CheckStatus.SKIPPED,
                    detail=(
                        f"multi-GPU topology present ({len(gpus)} GPUs); "
                        f"run validate-hardware --full for measured concurrent path"
                    ),
                    measured={"gpu_count": len(gpus), "full_probe": False},
                )
            )
    else:
        report.add(
            CheckResult(
                name="concurrent_gpus",
                status=CheckStatus.SKIPPED,
                detail="one GPU detected; multi-GPU validation requires at least two",
                measured={"gpu_count": 1, "full_probe": full},
            )
        )
    if cpus:
        if full:
            report.add(_try("concurrent_cpu_gpu", lambda: _probe_cpu_gpu_path(len(cpus), len(gpus))))
        else:
            report.add(
                CheckResult(
                    name="concurrent_cpu_gpu",
                    status=CheckStatus.SKIPPED,
                    detail=(
                        f"CPU+GPU topology present ({len(cpus)} NUMA pool(s), {len(gpus)} GPU(s)); "
                        f"run validate-hardware --full for measured path"
                    ),
                    measured={"cpu_pools": len(cpus), "gpu_count": len(gpus), "full_probe": False},
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


def _probe_multi_gpu_measured() -> CheckResult:
    import torch

    count = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    if count < 2:
        return CheckResult(
            name="concurrent_gpus",
            status=CheckStatus.SKIPPED,
            detail="full probe requested but fewer than two CUDA devices visible to torch",
            measured={"gpu_count": count},
        )
    a = torch.randn(256, 256, device="cuda:0")
    b = a.to("cuda:1")
    c = b.to("cuda:0")
    max_err = float((a - c).abs().max().item())
    ok = max_err < 1e-5
    p2p = False
    try:
        p2p = bool(torch.cuda.can_device_access_peer(0, 1))
    except Exception:  # noqa: BLE001
        p2p = False
    return CheckResult(
        name="concurrent_gpus",
        status=CheckStatus.CONCURRENT_EXECUTION_VALIDATED if ok else CheckStatus.FAILED,
        detail=f"measured H2D/D2D round-trip cuda:0↔cuda:1 max_abs_err={max_err:.3e} p2p={p2p}",
        measured={"gpu_count": count, "max_abs_err": max_err, "p2p": p2p, "full_probe": True},
    )


def _probe_cpu_gpu_path(cpu_pools: int, gpu_count: int) -> CheckResult:
    import torch
    import torch.nn as nn

    import tensortorrent as tt
    from tensortorrent.config import CompileConfig

    if not torch.cuda.is_available():
        return CheckResult(
            name="concurrent_cpu_gpu",
            status=CheckStatus.SKIPPED,
            detail="CUDA unavailable for measured CPU+GPU path",
            measured={"cpu_pools": cpu_pools, "gpu_count": gpu_count},
        )
    model = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 8)).eval()
    x = torch.randn(4, 128)
    with torch.no_grad():
        expected = model(x)
    compiled = tt.compile(
        model,
        (x,),
        config=CompileConfig(allow_cpu=True, allow_gpu=True, use_torch_compile=False),
    )
    try:
        out = compiled(x)
        max_err = float((out.detach().cpu() - expected).abs().max().item())
        devices = list(compiled.specialized.plan.devices_used)
        ok = max_err < 1e-4
        return CheckResult(
            name="concurrent_cpu_gpu",
            status=CheckStatus.CONCURRENT_EXECUTION_VALIDATED if ok else CheckStatus.FAILED,
            detail=f"measured compile path devices={devices} max_abs_err={max_err:.3e}",
            measured={
                "cpu_pools": cpu_pools,
                "gpu_count": gpu_count,
                "devices_used": devices,
                "max_abs_err": max_err,
                "full_probe": True,
            },
        )
    finally:
        compiled.close()


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
    """Compile a model with TensorTorrent and compare it against eager PyTorch.

    This check must execute the compiled path; comparing eager against eager
    would validate nothing.
    """
    try:
        import torch
        import torch.nn as nn

        import tensortorrent as tt
        from tensortorrent.validation.numerics import compare_module_outputs

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
        compiled = tt.compile(model, (x,))
        actual = compiled(x)
        numerics = compare_module_outputs(actual, expected)
        execution = compiled.last_execution_report()
        report.add(
            CheckResult(
                name="numerical_equivalence_eager",
                status=(CheckStatus.NUMERICAL_CORRECTNESS_VALIDATED if numerics.passed else CheckStatus.FAILED),
                detail=(
                    f"tensortorrent vs eager max_abs_err={numerics.max_abs_err:.3e} "
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
                    detail="run `tensortorrent autotune` on deployment hardware for measured plan comparison",
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


def _validate_stress(report: ValidationReport, *, overnight: bool = False) -> None:
    # Lightweight stability / leak probe suitable for CI and doctor --full.
    import gc

    import torch
    import torch.nn as nn

    import tensortorrent as tt

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

    # Bounded soak: short by default under --stress; overnight extends iterations.
    iters = 200 if overnight else 30
    wall_limit_s = 600.0 if overnight else 120.0
    rss_limit = 1024 * 1024 * 1024 if overnight else 512 * 1024 * 1024
    model = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 8)).eval()
    x = torch.randn(8, 64)
    compiled = tt.compile(model, (x,))
    try:
        soak_baseline = psutil_rss()
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(iters):
                _ = compiled(x)
        wall = time.perf_counter() - t0
        soak_delta = psutil_rss() - soak_baseline
        soak_ok = wall < wall_limit_s and soak_delta < rss_limit
        report.add(
            CheckResult(
                name="long_running_stability",
                status=CheckStatus.BASIC_EXECUTION_VALIDATED if soak_ok else CheckStatus.FAILED,
                detail=(
                    f"{'overnight' if overnight else 'short'} soak iters={iters} "
                    f"wall_s={wall:.3f} rss_delta={soak_delta}"
                ),
                measured={
                    "iters": iters,
                    "wall_s": wall,
                    "rss_delta": soak_delta,
                    "overnight": overnight,
                },
            )
        )
    except Exception as exc:  # noqa: BLE001
        report.add(
            CheckResult(
                name="long_running_stability",
                status=CheckStatus.FAILED,
                detail=str(exc),
            )
        )
    finally:
        compiled.close()


def psutil_rss() -> int:
    import os

    import psutil

    return int(psutil.Process(os.getpid()).memory_info().rss)


def _resolve_budgets_for_report() -> dict[str, Any]:
    """Return a JSON-serializable budgets section for ValidationReport.summary()."""
    from tensortorrent.hardware import budget as _budget

    result: dict[str, Any] = {}

    # Host memory
    try:
        hb = _budget.resolve_host_memory_budget()
        result["host_memory"] = {
            "total_bytes": hb.total_bytes,
            "allowed_bytes": hb.allowed_bytes,
            "reserved_bytes": hb.reserved_bytes,
            "source": hb.source.kind,
            "detail": hb.source.detail,
        }
    except Exception as exc:  # noqa: BLE001
        result["host_memory"] = {"error": str(exc)}

    # Effective CPUs
    try:
        cpu_count, cpu_src = _budget.resolve_cpu_budget()
        result["effective_cpus"] = {
            "count": cpu_count,
            "source": cpu_src.kind,
            "detail": cpu_src.detail,
        }
    except Exception as exc:  # noqa: BLE001
        result["effective_cpus"] = {"error": str(exc)}

    # Per-GPU VRAM
    gpus: dict[str, Any] = {}
    try:
        import torch

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                total = int(props.total_memory)
                free: int | None = None
                if hasattr(torch.cuda, "mem_get_info"):
                    try:
                        free_b, _ = torch.cuda.mem_get_info(i)
                        free = int(free_b)
                    except Exception:  # noqa: BLE001
                        pass
                from tensortorrent.backends.cuda import _probe_display_active

                display_active = _probe_display_active(i)
                headroom = _budget.default_vram_headroom_bytes(display_active)
                db = _budget.resolve_device_memory_budget(total, free, None, headroom)
                gpus[f"cuda_gpu_{i}"] = {
                    "total_bytes": db.total_bytes,
                    "allowed_bytes": db.allowed_bytes,
                    "reserved_bytes": db.reserved_bytes,
                    "source": db.source.kind,
                    "detail": db.source.detail,
                }
    except Exception as exc:  # noqa: BLE001
        gpus["error"] = str(exc)
    result["gpus"] = gpus

    # Spill disk
    try:
        import tempfile

        spill_path = str(tempfile.gettempdir())
        db = _budget.resolve_disk_budget(spill_path)
        result["spill_disk"] = {
            "path": spill_path,
            "total_bytes": db.total_bytes,
            "allowed_bytes": db.allowed_bytes,
            "reserved_bytes": db.reserved_bytes,
            "source": db.source.kind,
        }
    except Exception as exc:  # noqa: BLE001
        result["spill_disk"] = {"error": str(exc)}

    return result
