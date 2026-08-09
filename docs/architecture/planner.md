# Planner

Native Rust search (`crates/tt-planner`) shortlists strong placements. A discrete-event simulator then ranks concrete schedules for those finalists.

<p align="center">
  <img src="../figures/planner.svg" alt="TensorTorrent planner and DES flow" width="88%">
</p>

## Two stages

Full DES on every beam state would be too expensive. Search uses a cheap score; DES scores a bounded finalist set with contention, overlap, and residency. Analytical rank is not ground truth — DES can pick a non–rank-0 finalist, including another placement on the same devices.

## Planning problem

Specialization packs indexed native data before Rust: regions and deps, bytes, candidate kernels, measured/estimated latency, workspace, compute/memory resources, links, capacities, objective, and planner limits.

## Search

Eligible device subsets → bounded beam search → capacity/dominance pruning → optional local improvement. Multiple distinct terminals per competitive subset survive to the global merge (distinct by region/device/backend/kernel/dtype signature, not only the device set).

## Finalists and variants

- `planner_des_candidates` — max distinct placements sent to DES.
- `planner_per_subset_finalists` — survivors per subset before merge (`0` = automatic).

For streaming, a bounded prefetch-distance set is explored (preferred distance first, then neighbors). Pinned staging is preferred; pageable recovery can run if pinned fails capacity and fallback is allowed.

## Discrete-event simulation

DES consumes the real `ExecutableSchedule` and machine model: deps, compute availability, link contention, transfer cost, I/O, residency lifetime, peak memory, prefetch overlap, critical path. Kernel internals are not emulated instruction-by-instruction. Batch DES uses a bounded worker pool when profitable.

## Objectives

Finalists are ranked by `Objective`: `LATENCY`, `THROUGHPUT`, `MEMORY`, `BALANCED`, `WEIGHTED`. DES scores win; analytical/finalist rank and prefetch choice only break exact ties.

## Parallelism

Default is adaptive (`planner_workers=0`). `1` forces serial; `N` caps the Rayon pool. Subset search parallelizes first; large beams can expand intra-subset. Tiny workloads stay serial when pool overhead would dominate.

## Stats

Diagnostics keep stages separate: `analytic_rank`, `finalist_rank`, `simulated_rank`, plus worker/parallelism counters — so you can see when DES changed the winner.

## Non-goals

Does not exhaustively enumerate plans, force every accelerator, compile every finalist, use Python `ThreadPoolExecutor` as the primary planner pool, or hide transfer cost inside backend calls.
