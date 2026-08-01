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

import operator
from dataclasses import dataclass, field
from typing import Any

import torch
from torch.fx.passes.split_module import split_module
from torch.utils import _pytree as pytree

from streamcompiler.errors import GraphCaptureError, UnsupportedFeatureError

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
                "Input structure does not match the structure compiled by StreamCompiler.\n"
                f"  compiled with: {self.in_spec}\n"
                f"  called with:   {spec}\n"
                "Recompile with example inputs matching this call signature."
            )
        self._validate_inputs(flat, require_tensor=False)
        return flat

    def _validate_inputs(self, flat: list[Any], *, require_tensor: bool) -> None:
        """Reject inputs the compiled regions were not specialized for.

        StreamCompiler compiles static shapes, so a mismatch has to fail here with a
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
                    f"{tuple(value.shape)}. StreamCompiler compiles static shapes; recompile "
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
        """Unique parameter/buffer bytes (shared weights counted once)."""
        return self._unique_state_bytes(tuple(self.state_bindings))

    def max_region_state_bytes(self) -> int:
        """Largest parameter working set any single region needs at once."""
        return max(
            (self._unique_state_bytes(r.state_inputs) for r in self.regions),
            default=0,
        )

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


def _node_state_bytes(node: torch.fx.Node) -> int:
    """Bytes of parameter/buffer inputs this node reads."""
    total = 0
    for arg in node.all_input_nodes:
        if arg.op != "get_attr":
            continue
        spec = _value_spec(arg.name, arg.meta.get("val"), "parameter")
        total += spec.nbytes
    return total


def assign_partitions(
    graph: torch.fx.Graph,
    *,
    max_region_nodes: int = 16,
    max_region_state_bytes: int | None = None,
    force_single_region: bool = False,
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
    next_id = 0
    for node in graph.nodes:
        if node.op in _SOURCE_OPS or node.op == "output":
            continue
        node_state = _node_state_bytes(node)
        chosen: int | None = None
        producers = _node_producers(node)
        if len(producers) == 1:
            producer = producers[0]
            pid = partition.get(producer.name)
            fits_state = (
                max_region_state_bytes is None or state_bytes[pid] + node_state <= max_region_state_bytes
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
        state_bytes[chosen] += node_state
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
    original_meta = {node.name: node.meta.get("val") for node in module.graph.nodes}
    partition = assign_partitions(
        module.graph,
        max_region_nodes=max_region_nodes,
        max_region_state_bytes=max_region_state_bytes,
        force_single_region=force_single_region,
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
            "Exported graph contains no computation and returns no values; StreamCompiler cannot compile an empty model"
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
                "please report this model as a StreamCompiler bug"
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
