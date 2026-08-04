"""End-to-end proof that pack/stream/partition correctness fixes behave as intended."""

from __future__ import annotations

import operator

import torch
import torch.nn as nn
from torch.fx import symbolic_trace

import tensortorrent as tt
from tensortorrent.compile.regions import assign_partitions
from tensortorrent.runtime.schedule import OpCode
from tensortorrent.storage.pack import load_pack_manifest


class _Bf16MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


class _ChunkBranch(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.left = nn.Linear(8, 8)
        self.right = nn.Linear(8, 8)
        self.out = nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = torch.chunk(x, 2, dim=-1)
        return self.out(torch.relu(self.left(a) + self.right(b)))


class _SharedWeight(nn.Module):
    """Same Linear used twice so one pack target is consumed by two call sites."""

    def __init__(self) -> None:
        super().__init__()
        self.shared = nn.Linear(16, 16)
        self.head = nn.Linear(16, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.shared(x))
        x = torch.relu(self.shared(x))
        return self.head(x)


def test_bfloat16_streaming_compile_matches_eager() -> None:
    model = _Bf16MLP().to(dtype=torch.bfloat16).eval()
    x = torch.randn(4, 32, dtype=torch.bfloat16)
    with torch.no_grad():
        expected = model(x)
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    # One Linear fits; the full MLP does not → forces the streaming store.
    one_layer = model.fc1.weight.nbytes + model.fc1.bias.nbytes
    budget = one_layer + 256
    assert one_layer <= budget < total
    compiled = tt.compile(
        model,
        (x,),
        config=tt.CompileConfig(
            ram_budget_bytes=budget,
            max_region_nodes=1,
            allow_gpu=False,
            use_torch_compile=False,
            validate_numerics=True,
            atol=1e-2,
            rtol=1e-2,
        ),
    )
    try:
        store = compiled.executor.parameter_store
        assert store.stats()["kind"] == "streaming"
        pack_path = store._path
        manifest = load_pack_manifest(pack_path)
        dtypes = {t["stored_dtype"] for t in manifest["tensors"]}
        assert "bfloat16" in dtypes
        logical_ids = {t["logical_id"] for t in manifest["tensors"]}
        for env_name, target in compiled.program.state_bindings.items():
            assert target in logical_ids, f"missing pack key {target} for env {env_name}"
            assert env_name == target or env_name not in logical_ids
        with torch.no_grad():
            got = compiled(x)
        torch.testing.assert_close(got, expected, atol=1e-2, rtol=1e-2)
    finally:
        compiled.close()


def test_chunk_getitem_partition_and_compile_matches_eager() -> None:
    gm = symbolic_trace(_ChunkBranch())
    partition = assign_partitions(gm.graph, max_region_nodes=1)
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target is operator.getitem:
            producer = node.args[0]
            assert isinstance(producer, torch.fx.Node)
            assert partition[node.name] == partition[producer.name]

    model = _ChunkBranch().eval()
    x = torch.randn(3, 16)
    with torch.no_grad():
        expected = model(x)
    compiled = tt.compile(
        model,
        (x,),
        config=tt.CompileConfig(allow_gpu=False, use_torch_compile=False, max_region_nodes=1),
    )
    try:
        with torch.no_grad():
            got = compiled(x)
        torch.testing.assert_close(got, expected)
    finally:
        compiled.close()


def test_shared_weight_streaming_keeps_shared_until_last_use() -> None:
    model = _SharedWeight().eval()
    x = torch.randn(2, 16)
    with torch.no_grad():
        expected = model(x)
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    compiled = tt.compile(
        model,
        (x,),
        config=tt.CompileConfig(
            ram_budget_bytes=max(total // 2, 2048),
            allow_gpu=False,
            use_torch_compile=False,
            max_region_nodes=1,
            allow_concurrent_regions=True,
            max_concurrent_regions=2,
        ),
    )
    try:
        targets = list(compiled.program.state_bindings.values())
        shard_meta = compiled.program.metadata.get("linear_shards") or []
        reused_shards = [info for info in shard_meta if info.get("reused_state_shards")]
        # Shared Linear may remain as ``shared.weight`` or be rewritten into
        # exact output-feature shards that are reused across both call sites.
        assert any(t == "shared.weight" or t.endswith("shared.weight") for t in targets) or reused_shards, targets

        schedule = compiled.specialized.schedule
        assert schedule is not None
        shared_targets: set[str] = {tgt for tgt in targets if tgt == "shared.weight" or tgt.endswith("shared.weight")}
        for info in shard_meta:
            if info.get("weight_target") in {"shared.weight"} or str(info.get("weight_target", "")).endswith(
                "shared.weight"
            ):
                shared_targets.update(name for name in (info.get("shard_weights") or []) if name)
                shared_targets.update(name for name in (info.get("shard_biases") or []) if name)
        shared_envs = [env for env, tgt in compiled.program.state_bindings.items() if tgt in shared_targets]
        regions = list(compiled.program.regions)
        evicts = [
            inst
            for inst in schedule.instructions
            if inst.opcode == OpCode.EVICT and inst.attributes.get("kind") == "parameter_evict"
        ]
        if shared_envs and len(regions) > 1:
            region_state = {r.region_id: set(r.state_inputs) for r in regions}
            # Each shared env (full weight or individual shard) has its own last
            # use. Do not require every shard to stay pinned until the last shard
            # region finishes — only until that env's own final consumer.
            last_use: dict[str, str] = {}
            for region in regions:
                for env in set(shared_envs) & region_state[region.region_id]:
                    last_use[env] = region.region_id
            for inst in evicts:
                rid = inst.attributes.get("region_id")
                for env in set(inst.inputs) & set(shared_envs):
                    assert rid == last_use.get(env), (
                        f"region {rid} evicted shared state {env} before last use in {last_use.get(env)}"
                    )

        with torch.no_grad():
            got = compiled(x)
        torch.testing.assert_close(got, expected)

        store = compiled.executor.parameter_store
        if store.stats()["kind"] == "streaming":
            assert store.stats()["reads"] > 0
            assert store.stats()["peak_resident_bytes"] <= store.stats()["budget_bytes"]
    finally:
        compiled.close()


def test_pack_keys_match_streaming_bindings() -> None:
    model = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 4)).eval()
    x = torch.randn(2, 64)
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    one = model[0].weight.nbytes + model[0].bias.nbytes
    budget = one * 2
    assert budget < total
    compiled = tt.compile(
        model,
        (x,),
        config=tt.CompileConfig(
            ram_budget_bytes=budget,
            max_region_nodes=2,
            allow_gpu=False,
            use_torch_compile=False,
        ),
    )
    try:
        store = compiled.executor.parameter_store
        assert store.stats()["kind"] == "streaming"
        manifest = load_pack_manifest(store._path)
        logical_ids = {t["logical_id"] for t in manifest["tensors"]}
        # Streaming releases resident copies; pack keys must still resolve by target.
        for env_name, target in compiled.program.state_bindings.items():
            assert target in logical_ids, f"missing pack key {target} for env {env_name}; have {logical_ids}"
            tensor = store.acquire(env_name)
            assert tensor.numel() > 0
            store.release((env_name,))
        with torch.no_grad():
            expected = model(x)
            torch.testing.assert_close(compiled(x), expected)
    finally:
        compiled.close()
