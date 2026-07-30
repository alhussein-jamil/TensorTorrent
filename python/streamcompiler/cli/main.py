"""StreamCompiler CLI entrypoints for deployment-time operations."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from streamcompiler.compile.pipeline import (
    PortableArtifact,
    needs_respecialization,
    specialize_for_machine,
)
from streamcompiler.config import CompileConfig, Objective
from streamcompiler.hardware.discovery import discover_resource_graph, write_discovery_report
from streamcompiler.runtime.module import load_compiled
from streamcompiler.validation.hardware import validate_hardware


def _cmd_doctor(args: argparse.Namespace) -> int:
    report = validate_hardware(full=bool(args.full), stress=bool(args.full))
    print(report.render_text())
    if args.json:
        Path(args.json).write_text(json.dumps(report.summary(), indent=2), encoding="utf-8")
    failed = sum(1 for c in report.checks if c.status.value == "failed")
    return 1 if failed else 0


def _cmd_profile(args: argparse.Namespace) -> int:
    graph = discover_resource_graph()
    out = Path(args.output or "artifacts/profile")
    out.mkdir(parents=True, exist_ok=True)
    write_discovery_report(graph, out / "resource_graph.json")

    results: dict[str, object] = {"fingerprint": graph.fingerprint, "devices": {}, "transfers": {}}
    from streamcompiler.backends import available_backends
    from streamcompiler.cost_model import measure_host_copy

    # Host transfer baselines between NUMA / pinned pools when present.
    mem_names = list(graph.memory.keys())
    for src in mem_names[:4]:
        for dst in mem_names[:4]:
            if src == dst:
                continue
            model = measure_host_copy(src, dst, sizes=(1 << 20, 4 << 20))
            results["transfers"][f"{src}->{dst}"] = {  # type: ignore[index]
                "measured": model.measured,
                "alpha_s": model.alpha_s,
                "beta_bytes_per_s": model.beta_bytes_per_s,
                "samples": [{"nbytes": s.nbytes, "latency_s": s.latency_s} for s in model.samples],
            }

    from streamcompiler.cost_model import calibrate_host_priors

    results["host_priors"] = calibrate_host_priors()

    for backend in available_backends():
        sub = backend.discover_devices()
        for device in sub.compute.values():
            if args.all_resources or device.id.name in (args.devices or []):
                cands = backend.enumerate_kernels(
                    __import__("streamcompiler.ir.graph", fromlist=["Instruction"]).Instruction(
                        opcode=__import__("streamcompiler.ir.graph", fromlist=["OpCode"]).OpCode.COMPUTE,
                        name=f"profile_{device.id.name}",
                    ),
                    device,
                )
                device_results = []
                for cand in cands[:3]:
                    bench = backend.benchmark(cand)
                    device_results.append(
                        {
                            "kernel": cand.kernel_id,
                            "dtype": cand.dtype,
                            "latency_s": bench.latency_s,
                            "measured": bench.measured,
                            "notes": bench.notes,
                        }
                    )
                results["devices"][device.id.name] = device_results  # type: ignore[index]
    (out / "profile.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Storage benchmarks for discovered NVMe/disk resources.
    from streamcompiler.hardware.storage_bench import benchmark_storage_resources
    from streamcompiler.ir.resource_graph import MemoryClass

    mounts = []
    for mem in graph.memory.values():
        if mem.memory_class in (MemoryClass.NVME, MemoryClass.DISK_CACHE):
            mp = mem.attributes.get("mountpoint")
            if mp:
                mounts.append(mp)
    storage = benchmark_storage_resources(mounts[:4])
    results["storage"] = [
        {
            "path": s.path,
            "nbytes": s.nbytes,
            "latency_s": s.latency_s,
            "bytes_per_s": s.bytes_per_s,
            "measured": s.measured,
            "notes": s.notes,
        }
        for s in storage
    ]
    (out / "profile.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote profile to {out}")
    print(f"fingerprint={graph.fingerprint}")
    print(f"compute={len(graph.compute)} memory={len(graph.memory)} links={len(graph.links)}")
    print(f"transfer_models={len(results['transfers'])}")  # type: ignore[arg-type]
    print(f"storage_benchmarks={len(storage)}")
    return 0


def _cmd_validate_hardware(args: argparse.Namespace) -> int:
    report = validate_hardware(full=True, stress=bool(args.stress))
    print(report.render_text())
    out = Path(args.output or "artifacts/validation_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.summary(), indent=2), encoding="utf-8")
    print(f"wrote {out}")
    failed = sum(1 for c in report.checks if c.status.value == "failed")
    return 1 if failed else 0


def _cmd_benchmark_topology(args: argparse.Namespace) -> int:
    graph = discover_resource_graph()
    out = Path(args.output or "artifacts/topology.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = graph.summary()
    payload["independence_warnings"] = graph.validate_independence()
    # Explicit topology matrix for transfers.
    matrix = []
    for link in graph.links.values():
        matrix.append(
            {
                "name": link.id.name,
                "class": link.link_class.value,
                "source": link.source,
                "destination": link.destination,
                "p2p": link.peer_to_peer,
                "measured": link.measured,
                "bytes_per_s": link.bytes_per_s,
                "fallback": bool(link.attributes.get("fallback")),
            }
        )
    payload["transfer_matrix"] = matrix
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(out.read_text(encoding="utf-8"))
    return 0


def _cmd_autotune(args: argparse.Namespace) -> int:
    artifact_dir = Path(args.model_artifact)
    if not (artifact_dir / "portable.json").exists():
        print(f"error: portable artifact not found in {artifact_dir}", file=sys.stderr)
        return 2
    if needs_respecialization(artifact_dir) or args.force:
        print("specializing for current machine…")
    config = CompileConfig(
        objective=Objective(args.objective),
        profile_level="competitive" if args.profile else "coarse",
        allow_mixed_vendor=not args.no_mixed_vendor,
    )
    exported_path = artifact_dir / "exported.pt2"
    if exported_path.exists():
        # The exported program carries its example inputs, so this path can measure
        # regions on this machine instead of planning from priors.
        compiled = load_compiled(artifact_dir, config=config, refresh_artifacts=True)
        specialized = compiled.specialized
    else:
        print(
            "note: no exported.pt2 in the artifact; planning from priors only. "
            "Save with CompiledModule.save() to enable measured autotuning.",
            file=sys.stderr,
        )
        specialized = specialize_for_machine(
            PortableArtifact.load(artifact_dir),
            config=config,
            output_dir=artifact_dir / "specialized",
        )
    print(specialized.plan.explain())
    print(f"cached specialized artifact under {artifact_dir / 'specialized'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="streamcompiler",
        description="Heterogeneous streaming compiler for PyTorch inference",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="Diagnose hardware and backend readiness")
    doctor.add_argument("--full", action="store_true", help="Run extended validation probes")
    doctor.add_argument("--json", default="", help="Optional JSON report path")
    doctor.set_defaults(func=_cmd_doctor)

    profile = sub.add_parser("profile", help="Discover and benchmark machine resources")
    profile.add_argument("--all-resources", action="store_true", help="Profile every discovered resource")
    profile.add_argument("--devices", nargs="*", default=None)
    profile.add_argument("--output", default="artifacts/profile")
    profile.set_defaults(func=_cmd_profile)

    validate = sub.add_parser("validate-hardware", help="Production hardware validation suite")
    validate.add_argument("--stress", action="store_true")
    validate.add_argument("--output", default="artifacts/validation_report.json")
    validate.set_defaults(func=_cmd_validate_hardware)

    topo = sub.add_parser("benchmark-topology", help="Emit measured/discovered topology matrix")
    topo.add_argument("--output", default="artifacts/topology.json")
    topo.set_defaults(func=_cmd_benchmark_topology)

    autotune = sub.add_parser("autotune", help="Specialize a portable artifact for this machine")
    autotune.add_argument("model_artifact")
    autotune.add_argument("--objective", default="latency", choices=[o.value for o in Objective])
    autotune.add_argument("--profile", action="store_true")
    autotune.add_argument("--force", action="store_true")
    autotune.add_argument("--no-mixed-vendor", action="store_true")
    autotune.set_defaults(func=_cmd_autotune)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one CLI command and return its exit code.

    Returning instead of raising ``SystemExit`` keeps the commands callable from
    tests; the console-script wrapper turns the return value into the process code.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
