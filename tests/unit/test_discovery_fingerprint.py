"""Discovery stamps fingerprints used by the profiling cache."""

from __future__ import annotations

from streamcompiler.hardware.discovery import discover_resource_graph
from streamcompiler.ir.resource_graph import ComputeClass


def test_discovered_cpu_devices_carry_fingerprint_and_threads() -> None:
    graph = discover_resource_graph()
    assert graph.fingerprint
    cpus = [d for d in graph.compute.values() if d.compute_class == ComputeClass.CPU_NUMA_POOL]
    assert cpus
    for device in cpus:
        assert device.attributes.get("fingerprint")
        assert device.attributes.get("machine_fingerprint") == graph.fingerprint
        assert int(device.attributes.get("intraop_threads") or device.concurrency_limit or 0) > 0


def test_fingerprint_records_backend_plugin_identity(monkeypatch) -> None:
    import streamcompiler.hardware.fingerprint as fingerprint

    class Dist:
        name = "sample-backend"
        version = "1.2.3"

    class EntryPoint:
        name = "sample"
        value = "sample_backend:create"
        dist = Dist()

    class EntryPoints:
        def select(self, **kwargs):
            assert kwargs == {"group": "streamcompiler.backends"}
            return [EntryPoint()]

    monkeypatch.setattr(fingerprint.metadata, "entry_points", lambda: EntryPoints())
    monkeypatch.setattr(fingerprint, "_safe_run", lambda command: "")
    payload = fingerprint.collect_fingerprint_payload()
    assert payload["backend_plugins"] == [
        {
            "name": "sample",
            "value": "sample_backend:create",
            "distribution": "sample-backend",
            "version": "1.2.3",
        }
    ]
