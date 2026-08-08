# Configuration reference

`CompileConfig` controls portable compilation, machine specialization, and runtime policy. Defaults are chosen for general inference use; do not tune planner internals until profiling shows a reason.

```python
import tensortorrent as tt

config = tt.CompileConfig(objective=tt.Objective.LATENCY)
```

## Objectives

| Field | Default | Meaning |
| --- | --- | --- |
| `objective` | `LATENCY` | `latency`, `throughput`, `memory`, `balanced`, or `weighted` |
| `objective_weights` | latency 1, memory 0, throughput 0 | weights used by `WEIGHTED` |
| `target_inflight_requests` | `1` | concurrency assumption for throughput-oriented planning |

## Device policy

| Field | Default | Meaning |
| --- | ---: | --- |
| `allow_cpu` | `True` | allow CPU placements |
| `allow_gpu` | `True` | allow accelerator placements |
| `allow_integrated_gpu` | `True` | allow integrated GPU resources when discovered |
| `allow_mixed_vendor` | `True` | allow a plan to span accelerator vendors |
| `allow_host_staged_transfers` | `True` | permit host-staged fallback links |

Setting `allow_gpu=False` also disables integrated-GPU placement.

## Search and specialization

| Field | Default | Meaning |
| --- | ---: | --- |
| `max_plan_candidates` | `32` | upper-level candidate bound |
| `planner_beam_width` | `64` | non-dominated partial placements retained per step |
| `planner_candidates_per_device` | `2` | kernel/dtype candidates retained per device |
| `planner_local_search_iters` | `2` | bounded local-improvement passes |
| `planner_des_candidates` | `12` | distinct placement finalists considered by DES |
| `planner_per_subset_finalists` | `0` | per-subset terminal bound; `0` = automatic |
| `planner_parallel_subsets` | `True` | allow native subset-level parallel search |
| `planner_workers` | `0` | `0` auto, `1` serial, `>1` worker cap |
| `max_region_nodes` | `16` | maximum straight-line chain size per region |
| `region_compile_workers` | `1` | CPU region compile worker count; `0` = auto |
| `allow_concurrent_regions` | `True` | permit independent region execution overlap |
| `max_concurrent_regions` | `0` | `0` derives from selected devices |

Planner auto-parallelism stays serial on small searches when the native work estimate is below the threshold where threading is useful.

## Measurement and implementation selection

| Field | Default | Meaning |
| --- | ---: | --- |
| `measure_regions` | `True` | benchmark regions on real example tensors |
| `region_measure_iters` | `3` | measurement iterations |
| `measure_workers` | `0` | accelerator measurement workers; `0` = auto |
| `profile_level` | `"coarse"` | `coarse`, `competitive`, or `full` |
| `use_torch_compile` | `True` | allow TorchInductor candidate compilation |
| `torch_compile_backend` | `"inductor"` | backend passed to `torch.compile` |
| `prefer_direct_path` | `True` | use eligible low-overhead resident direct execution |
| `online_profile_feedback` | `True` | fold observed region latency into running priors |

`TT_DIRECT_PATH=0` forces schedule execution for otherwise eligible plans. `TT_DIRECT_PATH=1` forces attempting the eligible direct path.

## Memory, streaming, and storage

| Field | Default | Meaning |
| --- | ---: | --- |
| `ram_budget_bytes` | `None` | explicit host resident-parameter budget |
| `vram_budget_bytes` | `None` | explicit per-device accelerator memory cap |
| `activation_budget_bytes` | `None` | host activation budget used for spill planning |
| `allow_nvme_streaming` | `True` | permit parameter streaming from storage |
| `prefetch_distance` | `1` | minimum configured streaming prefetch distance |
| `adaptive_prefetch` | `True` | adapt prefetch using state/compute/budget information |
| `storage_io_workers` | `2` | native pack readers |
| `storage_queue_depth` | `128` | maximum outstanding native prefetch requests |
| `spill_dir` | `None` | activation spill root |
| `max_total_spill_bytes` | `None` | explicit total spill disk limit |
| `cache_dir` | `~/.cache/tensortorrent` | artifact/pack cache; `TT_CACHE_DIR` overrides default |

## Large operators and storage format

| Field | Default | Meaning |
| --- | ---: | --- |
| `enable_linear_sharding` | `True` | allow exact output-feature sharding of oversized linear ops |
| `max_linear_shards` | `128` | safety cap for generated shards |
| `allow_quantized_storage` | `False` | permit quantized storage representation |
| `numerical_mode` | `"exact"` | `exact` or `quantized` |

## Numerics

| Field | Default | Meaning |
| --- | ---: | --- |
| `validate_numerics` | `True` | compare specialized output with reference where applicable |
| `atol` | `1e-5` | absolute tolerance |
| `rtol` | `1e-5` | relative tolerance |

## Training and worker model

| Field | Default | Meaning |
| --- | ---: | --- |
| `allow_training` | `False` | opt in to schedule execution with autograd |
| `process_workers` | `0` | persistent Linux fork workers for selected CPU concurrency cases |

`allow_training=True` is incompatible with `process_workers>0` and activation spill. Process workers are not the normal accelerator execution model.

## Guardrails

| Field | Default | Meaning |
| --- | ---: | --- |
| `stall_timeout_s` | `300.0` | no-progress watchdog; `0` disables |
| `host_memory_reserve_bytes` | `None` | explicit host reserve override |
| `vram_headroom_bytes` | `None` | explicit per-GPU headroom override |
| `extra` | `{}` | reserved extension/config payload |

## Shared-desktop preset

```python
config = tt.CompileConfig.polite()
```

The preset increases VRAM headroom, limits concurrent regions, uses minimal prefetch, and shortens the stall threshold for a machine shared with interactive workloads.

## Environment overrides

Important environment variables include:

| Variable | Purpose |
| --- | --- |
| `TT_CACHE_DIR` | default cache root |
| `TT_SPILL_DIR` | default spill root |
| `TT_HOST_MEMORY_RESERVE_BYTES` | host reserve override |
| `TT_VRAM_HEADROOM_BYTES` | accelerator headroom override |
| `TT_DISABLE_VRAM_CAPACITY_FLOOR=1` | disable physical-capacity floor used to smooth transient allocator free-memory readings |
| `TT_DIRECT_PATH=0/1` | force/disable eligible direct execution behavior |
| `TENSORTORRENT_DISABLE_BACKEND_PLUGINS=1` | disable external backend entry points |

Serving-specific variables are documented in [Deployment](../product/deployment.md).
