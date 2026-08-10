"""Region program: executable partitioning of an exported PyTorch graph.

A :class:`RegionProgram` is the concrete, runnable object behind an execution
plan. It owns

* the real ``torch.fx`` subgraphs (one per region) produced by
  :func:`torch.fx.passes.split_module.split_module`,
* the dataflow edges between regions expressed as named environment values,
* the parameter/buffer/constant bindings each region needs,
* the pytree specs required to reproduce eager input and output structures.

The partitioning is hardware independent. Placement of regions onto devices is
the planner's job; this module only decides *what* the independently schedulable
units of work are.
"""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass, field
from typing import Any

import torch
from torch.fx.passes.split_module import split_module
from torch.utils import _pytree as pytree

from tensortorrent.errors import GraphCaptureError, UnsupportedFeatureError

#: FX node ops that produce a value without executing model computation.
_SOURCE_OPS = frozenset({"placeholder", "get_attr"})


def _is_tuple_getitem(node: torch.fx.Node) -> bool:
    """True when ``node`` indexes a multi-value producer (``operator.getitem``)."""
    return node.op == "call_function" and node.target is operator.getitem


@dataclass(frozen=True)
class ValueSpec:
    """Static description of one environment value (tensor or scalar)."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    nbytes: int
    kind: str  # input | parameter | buffer | constant | activation

    @property
    def is_tensor(self) -> bool:
        return self.dtype != "unknown"


@dataclass(frozen=True)
class Region:
    """One independently schedulable unit of real computation."""

    region_id: str
    submodule: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    multi_output: bool
    aten_ops: tuple[str, ...]
    node_count: int
    depends_on: tuple[str, ...]
    state_inputs: tuple[str, ...]
    output_bytes: int


@dataclass(frozen=True)
class RegionBinding:
    """A region together with the executable and device the planner chose for it."""

    region: Region
    compiled: Any
    backend_id: str
    device: str


@dataclass
class RegionProgram:
    """Executable form of a lowered model."""

    graph_name: str
    root: torch.nn.Module
    regions: tuple[Region, ...]
    user_inputs: tuple[str, ...]
    state_bindings: dict[str, str]
    values: dict[str, ValueSpec]
    output_refs: tuple[tuple[str, Any], ...]
    in_spec: pytree.TreeSpec
    out_spec: pytree.TreeSpec
    metadata: dict[str, Any] = field(default_factory=dict)
    # Fast-path descriptors, derived from the specs rather than assumed.
    _positional_tensor_arity: int = field(default=-1, init=False, repr=False)
    _single_output: bool = field(default=False, init=False, repr=False)
    _input_checks: tuple[tuple[str, tuple[int, ...], Any] | None, ...] = field(default=(), init=False, repr=False)

    def __post_init__(self) -> None:
        arity = len(self.user_inputs)
        sentinels = tuple(object() for _ in range(arity))
        if pytree.tree_structure((sentinels, {})) == self.in_spec:
            self._positional_tensor_arity = arity
        marker = object()
        try:
            self._single_output = pytree.tree_unflatten([marker], self.out_spec) is marker
        except ValueError:
            self._single_output = False
        checks: list[tuple[str, tuple[int, ...], Any] | None] = []
        for name in self.user_inputs:
            declared = self.values.get(name)
            if declared is None or not declared.is_tensor:
                checks.append(None)
                continue
            dtype = getattr(torch, declared.dtype, None)
            checks.append((name, declared.shape, dtype))
        self._input_checks = tuple(checks)

    def submodule(self, region: Region) -> torch.nn.Module:
        mod = getattr(self.root, region.submodule, None)
        if not isinstance(mod, torch.nn.Module):
            raise UnsupportedFeatureError(f"Region {region.region_id} has no submodule {region.submodule}")
        return mod

    def state_tensor(self, name: str) -> torch.Tensor:
        """Resolve one parameter/buffer/constant environment value."""
        target = self.state_bindings[name]
        obj: Any = self.root
        for part in target.split("."):
            obj = getattr(obj, part)
        return obj  # type: ignore[no-any-return]

    def state_tensors(self) -> dict[str, torch.Tensor]:
        """State values keyed by FX environment name (``linear_weight``)."""
        return {name: self.state_tensor(name) for name in self.state_bindings}

    def state_dict_for_pack(self) -> dict[str, torch.Tensor]:
        """Unique parameters/buffers keyed by module target path for packing.

        ``StreamingParameterStore`` resolves ``state_bindings`` by target
        (``linear.weight``), so packs must use the same key space.
        """
        packed: dict[str, torch.Tensor] = {}
        for env_name, target in self.state_bindings.items():
            if target not in packed:
                packed[target] = self.state_tensor(env_name)
        return packed

    @property
    def positional_tensor_arity(self) -> int:
        """Number of positional tensor args when the call signature is that simple, else -1."""
        return self._positional_tensor_arity

    @property
    def single_output(self) -> bool:
        """True when the compiled output pytree is a lone tensor (or other leaf)."""
        return self._single_output

    def flatten_inputs(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> list[Any]:
        """Flatten call arguments exactly the way ``torch.export`` recorded them."""
        if self._positional_tensor_arity == len(args) and not kwargs:
            for a in args:
                if type(a) is not torch.Tensor:
                    break
            else:
                flat = list(args)
                self._validate_inputs(flat, require_tensor=True)
                return flat
        flat, spec = pytree.tree_flatten((args, kwargs))
        if spec != self.in_spec:
            raise UnsupportedFeatureError(
                "Input structure does not match the structure compiled by TensorTorrent.\n"
                f"  compiled with: {self.in_spec}\n"
                f"  called with:   {spec}\n"
                "Recompile with example inputs matching this call signature."
            )
        self._validate_inputs(flat, require_tensor=False)
        return flat

    def _validate_inputs(self, flat: list[Any], *, require_tensor: bool) -> None:
        """Reject inputs the compiled regions were not specialized for.

        TensorTorrent compiles static shapes, so a mismatch has to fail here with a
        clear message; this check is what lets us drop ``torch.export``'s guard node.
        """
        if len(flat) != len(self.user_inputs):
            raise UnsupportedFeatureError(f"Expected {len(self.user_inputs)} flat inputs, received {len(flat)}")
        for name, value, check in zip(self.user_inputs, flat, self._input_checks, strict=True):
            if check is None:
                continue
            if not require_tensor and not isinstance(value, torch.Tensor):
                continue
            _, shape, dtype = check
            if value.shape != shape:
                raise UnsupportedFeatureError(
                    f"Input {name} was compiled for shape {shape} but received "
                    f"{tuple(value.shape)}. TensorTorrent compiles static shapes; recompile "
                    "for this shape."
                )
            if dtype is not None and value.dtype is not dtype:
                raise UnsupportedFeatureError(
                    f"Input {name} was compiled for dtype {str(dtype).removeprefix('torch.')} "
                    f"but received {_dtype_name(value)}"
                )

    def unflatten_outputs(self, flat: list[Any]) -> Any:
        if self._single_output and len(flat) == 1:
            return flat[0]
        return pytree.tree_unflatten(flat, self.out_spec)

    def region_by_id(self, region_id: str) -> Region:
        for region in self.regions:
            if region.region_id == region_id:
                return region
        raise KeyError(region_id)

    def execution_order(self) -> tuple[Region, ...]:
        return self.regions

    def total_state_bytes(self) -> int:
        """Unique parameter/buffer bytes (shared weights counted once).

        Export-free fused CPU programs intentionally keep ``state_bindings``
        empty (the original ``nn.Module`` owns the weights). Count those
        resident tensors from ``root`` so capacity accounting stays truthful.
        """
        total = self._unique_state_bytes(tuple(self.state_bindings))
        if total > 0:
            return total
        return self._export_free_root_state_bytes()

    def max_region_state_bytes(self) -> int:
        """Largest parameter working set any single region needs at once."""
        peak = max(
            (self._unique_state_bytes(r.state_inputs) for r in self.regions),
            default=0,
        )
        if peak > 0:
            return peak
        # Single fused region owns the whole resident root.
        return self._export_free_root_state_bytes()

    def _export_free_root_state_bytes(self) -> int:
        """Resident param/buffer bytes for export-free DirectPlan programs."""
        meta = self.metadata or {}
        if not isinstance(meta, dict) or not meta.get("eager_fused_export_free"):
            return 0
        root = self.root
        if not isinstance(root, torch.nn.Module):
            return 0
        return sum(int(t.numel()) * int(t.element_size()) for t in (*root.parameters(), *root.buffers()))

    def _unique_state_bytes(self, names: tuple[str, ...] | list[str]) -> int:
        """Sum nbytes once per underlying module attribute."""
        seen: set[str] = set()
        total = 0
        for name in names:
            target = self.state_bindings.get(name, name)
            if target in seen:
                continue
            seen.add(target)
            spec = self.values.get(name)
            if spec is not None:
                total += spec.nbytes
        return total

    def estimate_peak_activation_bytes(self) -> int:
        """Upper bound on live activation bytes under sequential last-use release."""
        consumers: dict[str, int] = {}
        for region in self.regions:
            for name in region.inputs:
                if name in self.state_bindings:
                    continue
                consumers[name] = consumers.get(name, 0) + 1
        for kind, ref in self.output_refs:
            if kind == "value":
                name = str(ref)
                consumers[name] = consumers.get(name, 0) + 1
        remaining = dict(consumers)
        live = 0
        peak = 0
        sizes = {name: spec.nbytes for name, spec in self.values.items() if name not in self.state_bindings}
        for region in self.regions:
            for name in region.outputs:
                live += sizes.get(name, 0)
            peak = max(peak, live)
            for name in region.inputs:
                if name in self.state_bindings or name not in remaining:
                    continue
                remaining[name] -= 1
                if remaining[name] == 0:
                    live -= sizes.get(name, 0)
                    remaining.pop(name, None)
        return peak


def _dtype_name(value: Any) -> str:
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        return "unknown"
    return str(dtype).replace("torch.", "")


def _value_spec(name: str, meta: Any, kind: str) -> ValueSpec:
    shape = tuple(int(x) for x in getattr(meta, "shape", ()) or ()) if hasattr(meta, "shape") else ()
    dtype = _dtype_name(meta)
    if dtype == "unknown":
        return ValueSpec(name=name, shape=(), dtype="unknown", nbytes=0, kind=kind)
    try:
        nbytes = int(torch.empty(0, dtype=getattr(torch, dtype)).element_size())
    except (AttributeError, TypeError):
        nbytes = 0
    for dim in shape:
        nbytes *= dim
    return ValueSpec(name=name, shape=shape, dtype=dtype, nbytes=nbytes, kind=kind)


def _node_producers(node: torch.fx.Node) -> list[torch.fx.Node]:
    producers: list[torch.fx.Node] = []
    for arg in node.all_input_nodes:
        if arg.op not in _SOURCE_OPS:
            producers.append(arg)
    return producers


def _resolve_get_attr_tensor(root: torch.nn.Module | None, node: torch.fx.Node) -> torch.Tensor | None:
    """Resolve a ``get_attr`` node to a live tensor on ``root`` when meta is empty."""
    if root is None or node.op != "get_attr":
        return None
    target = node.target
    if not isinstance(target, str):
        return None
    cur: Any = root
    for part in target.split("."):
        if not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    if torch.is_tensor(cur):
        return cur
    data = getattr(cur, "data", None)
    return data if torch.is_tensor(data) else None


@dataclass
class _StateBilling:
    """Charge each ``get_attr`` once per region when growing partition size caps."""

    root: torch.nn.Module | None
    billed: dict[int, set[str]] = field(default_factory=dict)

    def attr_nbytes(self, arg: torch.fx.Node) -> int:
        meta = arg.meta.get("val")
        spec = _value_spec(arg.name, meta, "parameter")
        if spec.nbytes > 0:
            return spec.nbytes
        tensor = _resolve_get_attr_tensor(self.root, arg)
        if tensor is None:
            return 0
        return int(tensor.numel()) * int(tensor.element_size())

    def incremental(self, pid: int, node: torch.fx.Node) -> int:
        already = self.billed.get(pid, set())
        add = 0
        seen_local: set[str] = set()
        for arg in node.all_input_nodes:
            if arg.op != "get_attr" or arg.name in already or arg.name in seen_local:
                continue
            seen_local.add(arg.name)
            add += self.attr_nbytes(arg)
        return add

    def charge(self, pid: int, node: torch.fx.Node) -> int:
        already = self.billed.setdefault(pid, set())
        add = 0
        for arg in node.all_input_nodes:
            if arg.op != "get_attr" or arg.name in already:
                continue
            already.add(arg.name)
            add += self.attr_nbytes(arg)
        return add


def _tensor_nbytes(value: Any) -> int:
    if not isinstance(value, torch.Tensor):
        return 0
    return int(value.numel()) * int(value.element_size())


def _register_shard_tensor(
    module: torch.nn.Module,
    name: str,
    value: torch.Tensor,
    *,
    parameter: bool,
) -> None:
    """Install one immutable shard as explicit lifted state on ``module``.

    Contiguous row slices already share the original storage. Keeping that view
    avoids a temporary second full parameter footprint during lowering; pack
    writing later materializes each exact slice with its storage offset applied.
    """
    detached = value.detach()
    if not detached.is_contiguous():
        detached = detached.contiguous()
    if parameter:
        module.register_parameter(name, torch.nn.Parameter(detached, requires_grad=False))
    else:
        module.register_buffer(name, detached, persistent=True)


def _get_or_create_get_attr(
    graph: torch.fx.Graph,
    cache: dict[str, torch.fx.Node],
    target: str,
) -> torch.fx.Node:
    """Return one get_attr node per attribute name (shared across tied uses)."""
    existing = cache.get(target)
    if existing is not None:
        return existing
    for node in graph.nodes:
        if isinstance(node, torch.fx.Node) and node.op == "get_attr" and str(node.target) == target:
            cache[target] = node
            return node
    created = graph.get_attr(target)
    if not isinstance(created, torch.fx.Node):  # pragma: no cover - fx API contract
        raise TypeError(f"expected get_attr node for {target!r}, got {type(created)!r}")
    cache[target] = created
    return created


def _cat_shard_pieces(
    payload: dict[str, Any],
    shard_names: list[str],
    *,
    gm_prefix: str,
    dim: int,
) -> torch.Tensor | None:
    pieces: list[torch.Tensor] = []
    for name in shard_names:
        value = payload.get(f"{gm_prefix}{name}")
        if not isinstance(value, torch.Tensor) or value.numel() == 0:
            return None
        pieces.append(value.detach())
    if len(pieces) != len(shard_names):
        return None
    return torch.cat(pieces, dim=dim)


def restore_sharded_state_dict(
    payload: dict[str, Any],
    linear_shards: list[dict[str, Any]],
    *,
    prefix: str = "",
) -> dict[str, Any]:
    """Replace linear-shard keys with reconstructed original parameter names.

    Only the first occurrence of a tied weight (``reused_state_shards=False``)
    rebuilds the full tensor. Shard keys are removed only after a successful
    concatenate so a partial failure never drops state.
    """
    gm_prefix = f"{prefix}graph_module."
    for info in linear_shards:
        if info.get("reused_state_shards"):
            continue
        weight_target = info.get("weight_target")
        bias_target = info.get("bias_target")
        shard_weights = [name for name in (info.get("shard_weights") or []) if name]
        shard_biases = [name for name in (info.get("shard_biases") or []) if name]
        if not weight_target or not shard_weights:
            continue

        weight = _cat_shard_pieces(payload, shard_weights, gm_prefix=gm_prefix, dim=0)
        if weight is None:
            continue
        payload[f"{gm_prefix}{weight_target}"] = weight
        for name in shard_weights:
            payload.pop(f"{gm_prefix}{name}", None)

        if bias_target and shard_biases:
            bias = _cat_shard_pieces(payload, shard_biases, gm_prefix=gm_prefix, dim=0)
            if bias is not None:
                payload[f"{gm_prefix}{bias_target}"] = bias
                for name in shard_biases:
                    payload.pop(f"{gm_prefix}{name}", None)
    return payload


def _shard_oversized_linear_nodes(
    module: torch.fx.GraphModule,
    *,
    max_region_state_bytes: int | None,
    max_linear_shards: int,
) -> list[dict[str, Any]]:
    """Rewrite oversized ``aten.linear`` into exact output-feature shards.

    A linear layer is separable along output features: each shard consumes the
    same activation and a disjoint row range of weight/bias, then the original
    result is reconstructed by concatenating shard outputs on the last axis.
    The rewrite therefore exposes genuine tensor-parallel regions without
    changing numerics or inventing collective semantics for arbitrary operators.
    """
    if max_region_state_bytes is None or max_region_state_bytes < 1:
        return []
    if max_linear_shards < 2:
        return []

    graph = module.graph
    rewritten: list[dict[str, Any]] = []
    shard_cache: dict[
        tuple[str, str | None, int],
        list[tuple[str, str | None, int, int]],
    ] = {}
    get_attr_cache: dict[str, torch.fx.Node] = {}
    linear_target = torch.ops.aten.linear.default
    for node in list(graph.nodes):
        if node.op != "call_function" or node.target != linear_target or node.kwargs:
            continue
        if len(node.args) != 3:
            continue
        activation, weight_node, bias_node = node.args
        if not isinstance(activation, torch.fx.Node) or not isinstance(weight_node, torch.fx.Node):
            continue
        if weight_node.op != "get_attr":
            continue
        if bias_node is not None and (not isinstance(bias_node, torch.fx.Node) or bias_node.op != "get_attr"):
            continue

        weight_target = str(weight_node.target)
        bias_target = str(bias_node.target) if isinstance(bias_node, torch.fx.Node) else None
        weight = _attr_value(module, weight_target)
        bias = _attr_value(module, bias_target) if bias_target is not None else None
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
            continue
        if bias is not None and (not isinstance(bias, torch.Tensor) or bias.ndim != 1):
            continue
        out_features = int(weight.shape[0])
        if out_features < 2:
            continue
        total_bytes = _tensor_nbytes(weight) + _tensor_nbytes(bias)
        if total_bytes <= max_region_state_bytes:
            continue

        row_bytes = _tensor_nbytes(weight[:1]) + (_tensor_nbytes(bias[:1]) if bias is not None else 0)
        if row_bytes <= 0 or row_bytes > max_region_state_bytes:
            # Even one mathematically valid row cannot fit the requested budget.
            # Leave the node intact so planning fails with an actionable capacity
            # error rather than silently violating the budget.
            continue
        rows_per_shard = max(1, int(max_region_state_bytes) // row_bytes)
        shard_count = int(math.ceil(out_features / rows_per_shard))
        if shard_count < 2 or shard_count > max_linear_shards:
            continue

        weight_is_parameter = isinstance(weight, torch.nn.Parameter)
        bias_is_parameter = isinstance(bias, torch.nn.Parameter)
        cache_key = (weight_target, bias_target, rows_per_shard)
        cached_shards = shard_cache.get(cache_key)
        shard_outputs: list[torch.fx.Node] = []
        ranges: list[tuple[int, int]] = []
        installed: list[tuple[str, str | None, int, int]] = []
        shard_weight_names: list[str] = []
        shard_bias_names: list[str | None] = []
        with graph.inserting_before(node):
            for shard_index, start in enumerate(range(0, out_features, rows_per_shard)):
                end = min(out_features, start + rows_per_shard)
                if cached_shards is not None:
                    weight_name, bias_name, cached_start, cached_end = cached_shards[shard_index]
                    if (cached_start, cached_end) != (start, end):  # pragma: no cover - cache key guarantees layout
                        raise RuntimeError("inconsistent tied linear shard layout")
                else:
                    prefix = f"_tt_{node.name}_shard_{shard_index}"
                    weight_name = f"{prefix}_weight"
                    bias_name = f"{prefix}_bias" if bias is not None else None
                    _register_shard_tensor(
                        module,
                        weight_name,
                        weight[start:end],
                        parameter=weight_is_parameter,
                    )
                    installed.append((weight_name, bias_name, start, end))
                shard_weight_names.append(weight_name)
                shard_bias_names.append(bias_name)

                # Tied / repeated linears must reuse the same get_attr nodes so
                # split_module lifts each shard target once. Fresh get_attrs per
                # use site would invent duplicate env bindings for one pack key.
                weight_get = _get_or_create_get_attr(graph, get_attr_cache, weight_name)
                weight_meta = weight_node.meta.get("val")
                if isinstance(weight_meta, torch.Tensor) and "val" not in weight_get.meta:
                    weight_get.meta["val"] = weight_meta[start:end]

                bias_get: torch.fx.Node | None = None
                if bias is not None:
                    assert bias_name is not None
                    if cached_shards is None:
                        _register_shard_tensor(
                            module,
                            bias_name,
                            bias[start:end],
                            parameter=bias_is_parameter,
                        )
                    bias_get = _get_or_create_get_attr(graph, get_attr_cache, bias_name)
                    bias_meta = bias_node.meta.get("val") if isinstance(bias_node, torch.fx.Node) else None
                    if isinstance(bias_meta, torch.Tensor) and "val" not in bias_get.meta:
                        bias_get.meta["val"] = bias_meta[start:end]

                shard = graph.call_function(linear_target, args=(activation, weight_get, bias_get))
                shard.meta.update(node.meta)
                output_meta = node.meta.get("val")
                if isinstance(output_meta, torch.Tensor):
                    shard.meta["val"] = output_meta[..., start:end]
                shard_outputs.append(shard)
                ranges.append((start, end))

            merged = graph.call_function(torch.ops.aten.cat.default, args=(shard_outputs, -1))
            merged.meta.update(node.meta)
        node.replace_all_uses_with(merged)
        graph.erase_node(node)
        if cached_shards is None:
            shard_cache[cache_key] = installed
        rewritten.append(
            {
                "node": node.name,
                "weight_target": weight_target,
                "bias_target": bias_target,
                "shard_weights": list(shard_weight_names),
                "shard_biases": list(shard_bias_names),
                "out_features": out_features,
                "original_state_bytes": total_bytes,
                "rows_per_shard": rows_per_shard,
                "shards": [list(bounds) for bounds in ranges],
                "reused_state_shards": cached_shards is not None,
            }
        )

    if rewritten:
        for node in list(graph.nodes):
            if node.op == "get_attr" and not node.users:
                graph.erase_node(node)
        graph.lint()
        module.recompile()
    return rewritten


def assign_partitions(
    graph: torch.fx.Graph,
    *,
    max_region_nodes: int = 16,
    max_region_state_bytes: int | None = None,
    force_single_region: bool = False,
    root: torch.nn.Module | None = None,
) -> dict[str, int]:
    """Split a graph into chain-shaped regions that break at branches and joins.

    A node joins its predecessor's region only when that predecessor is the
    region's current tail and feeds nothing else. Consequently:

    * straight-line chains collapse into one region (no scheduling overhead),
    * independent branches land in distinct regions (real parallelism),
    * join nodes start a new region (correct dependencies).

    ``max_region_nodes`` additionally caps chain length so long sequential models
    still expose several regions for pipelining across devices.

    ``max_region_state_bytes`` caps the weights a single region reads. Weight
    streaming needs this: a region's parameters must all be resident while it
    runs, so the streaming budget is only enforceable if regions are small enough
    to fit it.

    ``force_single_region`` puts every compute node in one region. Use it when
    concurrency measurement showed no benefit, so the runtime pays one subgraph
    call instead of one per branch.
    """
    if max_region_nodes < 1:
        raise ValueError("max_region_nodes must be >= 1")
    if force_single_region:
        return {node.name: 0 for node in graph.nodes if node.op not in _SOURCE_OPS and node.op != "output"}
    partition: dict[str, int] = {}
    tail: dict[int, str] = {}
    size: dict[int, int] = {}
    state_bytes: dict[int, int] = {}
    billing = _StateBilling(root)
    next_id = 0

    for node in graph.nodes:
        if node.op in _SOURCE_OPS or node.op == "output":
            continue
        chosen: int | None = None
        producers = _node_producers(node)
        if len(producers) == 1:
            producer = producers[0]
            pid = partition.get(producer.name)
            incremental = billing.incremental(pid, node) if pid is not None else 0
            fits_state = (
                max_region_state_bytes is None or state_bytes[pid] + incremental <= max_region_state_bytes
                if pid is not None
                else False
            )
            # Multi-value producers (``chunk`` / ``split`` / …) fan out to many
            # ``getitem`` users. Keep those getitems with the producer so
            # ``split_module`` never wires a single tensor into ``result[i]``.
            if (
                pid is not None
                and _is_tuple_getitem(node)
                or (
                    pid is not None
                    and len(producer.users) == 1
                    and tail.get(pid) == producer.name
                    and size[pid] < max_region_nodes
                    and fits_state
                )
            ):
                chosen = pid
        if chosen is None:
            chosen = next_id
            next_id += 1
            size[chosen] = 0
            state_bytes[chosen] = 0
        partition[node.name] = chosen
        tail[chosen] = node.name
        size[chosen] += 1
        state_bytes[chosen] += billing.charge(chosen, node)
    return partition


def _classify_state(root: torch.nn.Module) -> dict[str, str]:
    kinds: dict[str, str] = {}
    for name, _ in root.named_parameters():
        kinds[name] = "parameter"
    for name, _ in root.named_buffers():
        kinds[name] = "buffer"
    return kinds


def build_region_program(
    exported: Any,
    *,
    name: str = "model",
    max_region_nodes: int = 16,
    max_region_state_bytes: int | None = None,
    enable_linear_sharding: bool = True,
    max_linear_shards: int = 128,
    force_single_region: bool = False,
) -> RegionProgram:
    """Partition an ``ExportedProgram`` into executable regions."""
    try:
        module = exported.module()
    except Exception as exc:  # pragma: no cover - defensive
        raise GraphCaptureError(f"Cannot materialize exported module: {exc}") from exc

    graph_module = module if isinstance(module, torch.fx.GraphModule) else None
    if graph_module is None or not hasattr(module, "graph"):
        raise UnsupportedFeatureError("Exported module does not expose an fx graph")

    in_spec, out_spec = _call_specs(exported, module)
    dropped_guards = _drop_export_guards(module)
    dropped_device_asserts = _drop_device_metadata_asserts(module)
    linear_shards = (
        _shard_oversized_linear_nodes(
            module,
            max_region_state_bytes=max_region_state_bytes,
            max_linear_shards=max_linear_shards,
        )
        if enable_linear_sharding
        else []
    )
    original_meta = {node.name: node.meta.get("val") for node in module.graph.nodes}
    partition = assign_partitions(
        module.graph,
        max_region_nodes=max_region_nodes,
        max_region_state_bytes=max_region_state_bytes,
        force_single_region=force_single_region,
        root=module,
    )
    if partition:

        def split_callback(node: torch.fx.Node) -> int:
            return partition[node.name]

        root = split_module(module, module, split_callback, keep_original_order=True)
    elif _graph_returns_values(module.graph):
        # Pass-through graph: outputs are inputs, parameters or buffers, so there is
        # nothing to partition and the runtime resolves the outputs from its environment.
        root = module
    else:
        raise UnsupportedFeatureError(
            "Exported graph contains no computation and returns no values; TensorTorrent cannot compile an empty model"
        )
    state_kinds = _classify_state(root)

    values: dict[str, ValueSpec] = {}
    user_inputs: list[str] = []
    state_bindings: dict[str, str] = {}
    produced_by: dict[str, str] = {}
    regions: list[Region] = []
    output_refs: list[tuple[str, Any]] = []
    for node in root.graph.nodes:
        if node.op == "placeholder":
            values[node.name] = _value_spec(node.name, original_meta.get(node.name), "input")
            user_inputs.append(node.name)
        elif node.op == "get_attr":
            target = str(node.target)
            kind = state_kinds.get(target, "constant")
            values[node.name] = _value_spec(node.name, _attr_value(root, target), kind)
            state_bindings[node.name] = target
        elif node.op == "call_module":
            region_id = f"region_{len(regions)}"
            submodule = str(node.target)
            if any(not isinstance(a, torch.fx.Node) for a in node.args) or node.kwargs:
                raise UnsupportedFeatureError(f"Region {region_id} receives non-tensor graph arguments; unsupported")
            inputs = tuple(str(a.name) for a in node.args)
            sub = getattr(root, submodule)
            sub_outputs = _submodule_outputs(sub)
            multi_output = len(sub_outputs) > 1
            outputs = _region_output_names(node, sub_outputs, multi_output)
            aten_ops = tuple(
                _target_name(n.target) for n in sub.graph.nodes if n.op in ("call_function", "call_method")
            )
            node_count = sum(1 for n in sub.graph.nodes if n.op not in _SOURCE_OPS and n.op != "output")
            out_bytes = 0
            for out_name, out_node in zip(outputs, sub_outputs, strict=True):
                meta = out_node.meta.get("val") if isinstance(out_node, torch.fx.Node) else None
                spec = _value_spec(out_name, meta, "activation")
                values[out_name] = spec
                out_bytes += spec.nbytes
                produced_by[out_name] = region_id
            state_inputs = tuple(i for i in inputs if i in state_bindings)
            regions.append(
                Region(
                    region_id=region_id,
                    submodule=submodule,
                    inputs=inputs,
                    outputs=outputs,
                    multi_output=multi_output,
                    aten_ops=aten_ops,
                    node_count=node_count,
                    depends_on=(),
                    state_inputs=state_inputs,
                    output_bytes=out_bytes,
                )
            )
        elif node.op == "call_function" and node.target is operator.getitem:
            continue
        elif node.op == "output":
            flat_out = node.args[0]
            if not isinstance(flat_out, (tuple, list)):
                flat_out = (flat_out,)
            for item in flat_out:
                if isinstance(item, torch.fx.Node):
                    output_refs.append(("value", str(item.name)))
                else:
                    output_refs.append(("constant", item))
        elif node.op == "call_function":
            raise UnsupportedFeatureError(
                f"Unexpected top-level operation {_target_name(node.target)} after partitioning; "
                "please report this model as a TensorTorrent bug"
            )
        else:  # pragma: no cover - split_module produces no other ops
            raise UnsupportedFeatureError(f"Unsupported top-level fx op {node.op}")

    resolved: list[Region] = []
    for region in regions:
        deps = tuple(
            dict.fromkeys(
                produced_by[i] for i in region.inputs if i in produced_by and produced_by[i] != region.region_id
            )
        )
        resolved.append(
            Region(
                region_id=region.region_id,
                submodule=region.submodule,
                inputs=region.inputs,
                outputs=region.outputs,
                multi_output=region.multi_output,
                aten_ops=region.aten_ops,
                node_count=region.node_count,
                depends_on=deps,
                state_inputs=region.state_inputs,
                output_bytes=region.output_bytes,
            )
        )

    missing = [ref for kind, ref in output_refs if kind == "value" and ref not in values]
    if missing:
        raise UnsupportedFeatureError(f"Graph outputs reference unknown values: {missing}")

    return RegionProgram(
        graph_name=name,
        root=root,
        regions=tuple(resolved),
        user_inputs=tuple(user_inputs),
        state_bindings=state_bindings,
        values=values,
        output_refs=tuple(output_refs),
        in_spec=in_spec,
        out_spec=out_spec,
        metadata={
            "max_region_nodes": max_region_nodes,
            "max_region_state_bytes": max_region_state_bytes,
            "force_single_region": force_single_region,
            "region_count": len(resolved),
            "state_value_count": len(state_bindings),
            "export_guards_removed": dropped_guards,
            "device_metadata_asserts_removed": dropped_device_asserts,
            "linear_shards": linear_shards,
        },
    )


def _graph_returns_values(graph: torch.fx.Graph) -> bool:
    for node in graph.nodes:
        if node.op == "output":
            flat, _ = pytree.tree_flatten(node.args[0])
            return any(item is not None for item in flat)
    return False


def _drop_export_guards(module: torch.fx.GraphModule) -> int:
    """Remove ``torch.export``'s input-guard node from our copy of the graph.

    The guard re-checks input shapes on every call. :meth:`RegionProgram.flatten_inputs`
    already validates the shape and dtype of every input against what was compiled,
    with a clearer error, so keeping the guard only adds per-call Python work.
    """
    removed = 0
    for node in list(module.graph.nodes):
        if node.op != "call_module" or node.users:
            continue
        target = str(node.target)
        submodule = getattr(module, target, None)
        is_guard = target.startswith("_guards") or "guard" in type(submodule).__name__.lower()
        if not is_guard:
            continue
        module.graph.erase_node(node)
        removed += 1
    if removed:
        module.recompile()
    return removed


def _is_assert_tensor_metadata(target: Any) -> bool:
    name = getattr(target, "__name__", None) or getattr(target, "_opname", None) or str(target)
    return "assert_tensor_metadata" in str(name)


def _drop_device_metadata_asserts(module: torch.fx.GraphModule) -> int:
    """Remove export-time ``_assert_tensor_metadata`` nodes.

    ``torch.export`` on CPU embeds ``device=cpu`` checks. Under schedule-managed
    placement those tensors move to accelerators before compute, so the assert
    raises ``Expected: cpu, Got: cuda:0``. Shape/dtype are already enforced by
    :meth:`RegionProgram.flatten_inputs`; device residency is owned by the
    schedule. Replace uses with the asserted tensor (passthrough).

    Walks nested ``GraphModule`` children (HOPs like ``wrap_with_set_grad_enabled``).
    """
    removed = 0
    for _child_name, child in list(module.named_children()):
        if isinstance(child, torch.fx.GraphModule):
            removed += _drop_device_metadata_asserts(child)
    for node in list(module.graph.nodes):
        if node.op != "call_function" or not _is_assert_tensor_metadata(node.target):
            continue
        tensor_arg = node.args[0] if node.args else None
        node.replace_all_uses_with(tensor_arg)
        module.graph.erase_node(node)
        removed += 1
    if removed:
        module.graph.lint()
        module.recompile()
    return removed


def _submodule_outputs(sub: torch.fx.GraphModule) -> tuple[Any, ...]:
    """Return the fx nodes (or literals) a split submodule returns, in order."""
    for node in sub.graph.nodes:
        if node.op != "output":
            continue
        result = node.args[0]
        if isinstance(result, (tuple, list)):
            return tuple(result)
        return (result,)
    raise UnsupportedFeatureError("Split submodule has no output node")


def _region_output_names(
    call_node: torch.fx.Node,
    sub_outputs: tuple[Any, ...],
    multi_output: bool,
) -> tuple[str, ...]:
    """Name each region result, reusing downstream ``getitem`` node names."""
    if not sub_outputs:
        # Side-effect-only region, e.g. the shape guards torch.export inserts.
        return ()
    if not multi_output:
        if any(u.op == "call_function" and u.target is operator.getitem for u in call_node.users):
            raise UnsupportedFeatureError(f"Region {call_node.name} is indexed but returns one value")
        return (str(call_node.name),)
    by_index: dict[int, str] = {}
    for user in call_node.users:
        if user.op != "call_function" or user.target is not operator.getitem:
            raise UnsupportedFeatureError(f"Region {call_node.name} returns a tuple consumed by a non-index operation")
        index = user.args[1]
        if not isinstance(index, int):
            raise UnsupportedFeatureError(f"Region {call_node.name} is indexed by a non-constant {index!r}")
        by_index[index] = str(user.name)
    return tuple(by_index.get(i, f"{call_node.name}_out{i}") for i in range(len(sub_outputs)))


def _attr_value(root: torch.nn.Module, target: str) -> Any:
    obj: Any = root
    for part in target.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def _target_name(target: Any) -> str:
    name = getattr(target, "_opname", None)
    if name is not None:
        return f"aten::{name}"
    return getattr(target, "__name__", str(target))


def _call_specs(exported: Any, module: Any) -> tuple[pytree.TreeSpec, pytree.TreeSpec]:
    call_spec = getattr(exported, "call_spec", None)
    in_spec = getattr(call_spec, "in_spec", None)
    out_spec = getattr(call_spec, "out_spec", None)
    if in_spec is None:
        in_spec = getattr(module, "_in_spec", None)
    if out_spec is None:
        out_spec = getattr(module, "_out_spec", None)
    if in_spec is None or out_spec is None:
        raise UnsupportedFeatureError("Exported program does not expose pytree call specs")
    return in_spec, out_spec
