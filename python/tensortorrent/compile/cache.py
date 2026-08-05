"""Artifact cache and fingerprint helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tensortorrent.compile.artifacts import SpecializedArtifact
from tensortorrent.hardware.fingerprint import machine_fingerprint


def _attach_storage_measurement(store: Any, specialized: SpecializedArtifact) -> None:
    """Record measured pack pread bandwidth when the runtime streams from disk."""
    if getattr(store, "kind", None) != "streaming":
        return
    from tensortorrent.hardware.storage_bench import benchmark_pack_payload
    from tensortorrent.storage.pack import load_pack_manifest

    stats = store.stats()
    pack_path = Path(stats["pack_path"])
    manifest = load_pack_manifest(pack_path)
    tensors = manifest.get("tensors") or []
    if not tensors:
        return
    largest = max(tensors, key=lambda entry: int(entry.get("nbytes", 0)))
    result = benchmark_pack_payload(
        pack_path,
        offset=int(largest["offset"]),
        nbytes=int(largest["nbytes"]),
    )
    specialized.profile["storage"] = result.as_dict()
    specialized.validation["storage"] = result.as_dict()
    if result.measured:
        mbps = result.bytes_per_s / (1 << 20)
        specialized.plan.notes.append(
            f"storage_pread_measured={mbps:.1f} MiB/s "
            f"({result.nbytes} bytes in {result.latency_s * 1e3:.3f} ms; {result.notes})"
        )
    else:
        specialized.plan.notes.append(f"storage_pread_unmeasured: {result.notes}")


def needs_respecialization(artifact_dir: Path, current_fingerprint: str | None = None) -> bool:
    """True when no matching fingerprint exists for this machine.

    Looks at both the artifact root and ``specialized/fingerprint`` because
    ``SpecializedArtifact.save`` writes under ``specialized/`` while
    ``CompiledModule.save`` also mirrors the fingerprint at the root.
    """
    current = current_fingerprint or machine_fingerprint()
    for relative in ("fingerprint", "specialized/fingerprint"):
        fp_path = artifact_dir / relative
        if not fp_path.exists():
            continue
        stored = fp_path.read_text(encoding="utf-8").strip()
        return stored != current
    return True
