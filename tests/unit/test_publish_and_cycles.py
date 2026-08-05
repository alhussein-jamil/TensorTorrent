"""ExecutionContext publish API keeps CopyStore + native mirror paired."""

from __future__ import annotations

from tensortorrent.runtime.execution_context import ExecutionContext


def test_publish_tensor_updates_copy_store_without_native() -> None:
    ctx = ExecutionContext(host_resource="cpu")
    value = object()
    copy = ctx.publish_tensor("t0", "cpu", value, ownership="activation")
    assert copy.value is value
    assert ctx.copies.require("t0", "cpu").value is value


def test_publish_replica_does_not_drop_source() -> None:
    ctx = ExecutionContext(host_resource="cpu")
    value = object()
    ctx.publish_tensor("t0", "cpu", value, ownership="activation")
    ctx.publish_replica("t0", "cpu_b", value, ownership="transfer", source_resource="cpu")
    assert ctx.copies.has("t0", "cpu")
    assert ctx.copies.has("t0", "cpu_b")
    assert set(ctx.copies.resources_for("t0")) == {"cpu", "cpu_b"}


def test_schedule_executor_does_not_import_graph_executor() -> None:
    """Cycle break: schedule_executor must not pull graph_executor."""
    from pathlib import Path

    import tensortorrent.runtime.schedule_executor as se

    source = Path(se.__file__).read_text(encoding="utf-8")
    assert "graph_executor" not in source
    assert "fork_regions" in source
