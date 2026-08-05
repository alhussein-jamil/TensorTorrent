"""ExecutionContext publish API keeps CopyStore + native mirror paired."""

from __future__ import annotations

import pytest

from tensortorrent.errors import RuntimePlanError
from tensortorrent.runtime.execution_context import ExecutionContext
from tensortorrent.runtime.handles import NativeResidencyBridge


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


def test_attach_native_blocks_direct_copy_store_put() -> None:
    ctx = ExecutionContext(host_resource="cpu")
    bridge = NativeResidencyBridge.create()
    ctx.attach_native_residency(bridge)
    with pytest.raises(RuntimePlanError, match="publish_tensor"):
        ctx.copies.put("t0", "cpu", object())


def test_alias_and_drop_copy_under_native() -> None:
    ctx = ExecutionContext(host_resource="cpu")
    bridge = NativeResidencyBridge.create()
    ctx.attach_native_residency(bridge)
    value = object()
    ctx.publish_tensor("t0", "cpu", value, ownership="input")
    ctx.alias_copy("t0", "cpu", "host")
    assert ctx.copies.has("t0", "host")
    assert bridge.session.has("t0", "host")
    ctx.drop_copy("t0", "host", rust_already_released=False)
    assert not ctx.copies.has("t0", "host")


def test_schedule_executor_catchup_uses_publish_not_raw_put() -> None:
    """Transfer→Compute catch-up must go through publish_* (no raw copies.put)."""
    from pathlib import Path

    import tensortorrent.runtime.schedule_executor as se

    source = Path(se.__file__).read_text(encoding="utf-8")
    assert "ctx.copies.put(" not in source
    assert "ctx.copies.replicate(" not in source
    assert "publish_replica" in source
    assert "publish_tensor" in source
