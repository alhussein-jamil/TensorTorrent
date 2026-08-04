"""Exact output-feature sharding for oversized linear operators."""

from __future__ import annotations

import torch

from tensortorrent.compile.pipeline import _region_state_budget
from tensortorrent.compile.regions import restore_sharded_state_dict
from tensortorrent.config import CompileConfig
from tensortorrent.frontend.lower import lower_exported_program
from tensortorrent.ir.resource_graph import (
    ComputeClass,
    ComputeResource,
    MemoryClass,
    MemoryResource,
    ResourceGraph,
    ResourceId,
    ResourceKind,
)


class _Linear(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(8, 12)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class _TiedLinear(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(12, 8))
        self.bias = torch.nn.Parameter(torch.randn(12))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.weight, self.bias) + torch.nn.functional.linear(
            x, self.weight, self.bias
        )


def _run_program(program: object, x: torch.Tensor) -> torch.Tensor:
    env = dict(zip(program.user_inputs, [x], strict=True))
    env.update(program.state_tensors())
    for region in program.execution_order():
        result = program.submodule(region)(*(env[name] for name in region.inputs))
        outputs = tuple(result) if region.multi_output else (result,)
        for name, value in zip(region.outputs, outputs, strict=True):
            env[name] = value
    flat = [env[str(ref)] if kind == "value" else ref for kind, ref in program.output_refs]
    return program.unflatten_outputs(flat)


def test_oversized_linear_is_sharded_and_reconstructed_exactly() -> None:
    torch.manual_seed(4)
    model = _Linear().eval()
    x = torch.randn(3, 8)
    exported = torch.export.export(model, (x,))

    lowered = lower_exported_program(
        exported,
        name="linear",
        max_region_state_bytes=160,
        enable_linear_sharding=True,
        max_linear_shards=16,
    )
    program = lowered.program

    shards = program.metadata["linear_shards"]
    assert len(shards) == 1
    assert len(shards[0]["shards"]) == 3
    assert program.max_region_state_bytes() <= 160
    assert sum("aten::linear" in region.aten_ops for region in program.regions) == 3
    assert any("aten::cat" in region.aten_ops for region in program.regions)
    # Shards are views over one weight storage and one bias storage, not cloned
    # copies that double host memory during lowering.
    storages = {tensor.untyped_storage().data_ptr() for tensor in program.state_tensors().values()}
    assert len(storages) == 2
    torch.testing.assert_close(_run_program(program, x), model(x), rtol=0, atol=0)


def test_linear_sharding_can_be_disabled() -> None:
    model = _Linear().eval()
    x = torch.randn(1, 8)
    exported = torch.export.export(model, (x,))
    lowered = lower_exported_program(
        exported,
        max_region_state_bytes=160,
        enable_linear_sharding=False,
    )
    assert lowered.program.metadata["linear_shards"] == []
    assert sum("aten::linear" in region.aten_ops for region in lowered.program.regions) == 1


def test_tied_linear_weights_reuse_storage_shards() -> None:
    model = _TiedLinear().eval()
    x = torch.randn(2, 8)
    exported = torch.export.export(model, (x,))
    lowered = lower_exported_program(
        exported,
        max_region_state_bytes=160,
        enable_linear_sharding=True,
    )
    program = lowered.program
    metadata = program.metadata["linear_shards"]
    assert len(metadata) == 2
    assert metadata[0]["reused_state_shards"] is False
    assert metadata[1]["reused_state_shards"] is True
    assert metadata[0]["weight_target"] == "weight"
    assert metadata[0]["shard_weights"]
    # Tied uses must share get_attr env bindings (one pack key per shard).
    assert len(set(program.state_bindings.values())) == len(program.state_bindings)
    # Two uses of tied state must not duplicate packed parameter bytes.
    assert program.total_state_bytes() == model.weight.nbytes + model.bias.nbytes
    torch.testing.assert_close(_run_program(program, x), model(x), rtol=0, atol=0)


def test_restore_sharded_state_dict_rebuilds_original_names() -> None:
    w0 = torch.randn(2, 4)
    w1 = torch.randn(3, 4)
    b0 = torch.randn(2)
    b1 = torch.randn(3)
    payload = {
        "graph_module._tt_w0": w0,
        "graph_module._tt_w1": w1,
        "graph_module._tt_b0": b0,
        "graph_module._tt_b1": b1,
        "graph_module.head.weight": torch.ones(1, 1),
    }
    meta = [
        {
            "weight_target": "linear.weight",
            "bias_target": "linear.bias",
            "shard_weights": ["_tt_w0", "_tt_w1"],
            "shard_biases": ["_tt_b0", "_tt_b1"],
            "reused_state_shards": False,
        },
        {
            "weight_target": "linear.weight",
            "bias_target": "linear.bias",
            "shard_weights": ["_tt_w0", "_tt_w1"],
            "shard_biases": ["_tt_b0", "_tt_b1"],
            "reused_state_shards": True,
        },
    ]
    out = restore_sharded_state_dict(payload, meta)
    torch.testing.assert_close(out["graph_module.linear.weight"], torch.cat([w0, w1], dim=0))
    torch.testing.assert_close(out["graph_module.linear.bias"], torch.cat([b0, b1], dim=0))
    assert "graph_module._tt_w0" not in out
    assert "graph_module._tt_b1" not in out
    assert "graph_module.head.weight" in out


def test_restore_sharded_state_dict_keeps_shards_on_partial_failure() -> None:
    bad_weight = {
        "graph_module._tt_w0": torch.randn(2, 4),
        "graph_module._tt_w1": torch.zeros(0),  # empty placeholder / missing data
        "graph_module._tt_b0": torch.randn(2),
        "graph_module._tt_b1": torch.randn(2),
    }
    meta = [
        {
            "weight_target": "linear.weight",
            "bias_target": "linear.bias",
            "shard_weights": ["_tt_w0", "_tt_w1"],
            "shard_biases": ["_tt_b0", "_tt_b1"],
            "reused_state_shards": False,
        }
    ]
    out = restore_sharded_state_dict(bad_weight, meta)
    assert "graph_module.linear.weight" not in out
    assert "graph_module._tt_w0" in out
    # Bias must not be rebuilt or stripped when weight reconstruction fails.
    assert "graph_module.linear.bias" not in out
    assert "graph_module._tt_b0" in out

    bad_bias = {
        "graph_module._tt_w0": torch.randn(2, 4),
        "graph_module._tt_w1": torch.randn(2, 4),
        "graph_module._tt_b0": torch.randn(2),
        "graph_module._tt_b1": torch.zeros(0),
    }
    out = restore_sharded_state_dict(bad_bias, meta)
    assert "graph_module.linear.weight" in out
    assert "graph_module._tt_w0" not in out
    assert "graph_module.linear.bias" not in out
    assert "graph_module._tt_b0" in out


def test_region_budget_uses_smallest_eligible_accelerator_with_headroom() -> None:
    machine = ResourceGraph(fingerprint="shard-budget")
    for index, capacity in enumerate((1_000, 2_000)):
        device = f"gpu_{index}"
        memory = f"vram_{index}"
        machine.add_memory(
            MemoryResource(
                id=ResourceId(ResourceKind.MEMORY, memory),
                memory_class=MemoryClass.DEVICE_VRAM,
                capacity_bytes=capacity,
                allocatable_bytes=capacity,
                attached_compute=(device,),
            )
        )
        machine.add_compute(
            ComputeResource(
                id=ResourceId(ResourceKind.COMPUTE, device),
                compute_class=ComputeClass.DISCRETE_GPU,
                backend_id="cuda",
                model=device,
                memory_affinity=(memory,),
            )
        )
    assert _region_state_budget(CompileConfig(), machine) == 700
    assert _region_state_budget(CompileConfig(vram_budget_bytes=600), machine) == 420
    # Host streaming can impose a stricter cap than VRAM.
    assert _region_state_budget(CompileConfig(ram_budget_bytes=400, prefetch_distance=1), machine) == 200
