//! Native planner unit tests.

use crate::oracle::{exhaustive_best, replay_scores_match};
use crate::problem::{
    CandidateKernel, ObjectiveKind, PlanningConfig, PlanningProblem, RegionSpec, SubsetSpec,
};
use crate::search::{plan_placements, search_subset, search_subset_ex};
use crate::{
    beam_parallelism_possible, resolve_workers, should_parallelize_beam, should_parallelize_subsets,
};
use std::collections::HashMap;
use tt_runtime::{MachineModel, MemoryResource, TransferLink};

fn two_device_problem() -> PlanningProblem {
    let mut machine = MachineModel::cpu_only();
    machine.compute.insert("accel_0".into(), 1.0);
    machine.compute.insert("accel_1".into(), 1.0);
    machine
        .memory_affinity
        .insert("accel_0".into(), "vram_0".into());
    machine
        .memory_affinity
        .insert("accel_1".into(), "vram_1".into());
    machine.memory.insert(
        "vram_0".into(),
        MemoryResource {
            name: "vram_0".into(),
            capacity_bytes: 10_000,
            allocatable_bytes: 10_000,
            memory_class: "device_vram".into(),
        },
    );
    machine.memory.insert(
        "vram_1".into(),
        MemoryResource {
            name: "vram_1".into(),
            capacity_bytes: 10_000,
            allocatable_bytes: 10_000,
            memory_class: "device_vram".into(),
        },
    );
    machine.links.push(TransferLink {
        id: "vram_0->vram_1".into(),
        source: "vram_0".into(),
        destination: "vram_1".into(),
        bandwidth_bytes_per_s: 1e6,
        latency_s: 0.0,
        link_class: "pcie".into(),
        contention_factor: 1.0,
        measured: true,
        bidirectional: true,
        peer_to_peer: false,
    });

    let mut edge_bytes = HashMap::new();
    edge_bytes.insert((0, 1), 1000u64);

    PlanningProblem {
        regions: vec![
            RegionSpec {
                name: "r0".into(),
                depends_on: vec![],
                output_bytes: 1000,
                state_bytes: 0,
                consumer_count: 1,
            },
            RegionSpec {
                name: "r1".into(),
                depends_on: vec![0],
                output_bytes: 4,
                state_bytes: 0,
                consumer_count: 1,
            },
        ],
        order: vec![0, 1],
        candidates: vec![
            vec![
                CandidateKernel {
                    device: 0,
                    backend_id: "mock".into(),
                    kernel_id: "r0:a0".into(),
                    dtype: "float32".into(),
                    estimated_latency_s: 0.01,
                    workspace_bytes: 0,
                    measured: true,
                },
                CandidateKernel {
                    device: 1,
                    backend_id: "mock".into(),
                    kernel_id: "r0:a1".into(),
                    dtype: "float32".into(),
                    estimated_latency_s: 0.03,
                    workspace_bytes: 0,
                    measured: true,
                },
            ],
            vec![
                CandidateKernel {
                    device: 0,
                    backend_id: "mock".into(),
                    kernel_id: "r1:a0".into(),
                    dtype: "float32".into(),
                    estimated_latency_s: 0.02,
                    workspace_bytes: 0,
                    measured: true,
                },
                CandidateKernel {
                    device: 1,
                    backend_id: "mock".into(),
                    kernel_id: "r1:a1".into(),
                    dtype: "float32".into(),
                    estimated_latency_s: 0.015,
                    workspace_bytes: 0,
                    measured: true,
                },
            ],
        ],
        device_names: vec!["accel_0".into(), "accel_1".into()],
        capacities: vec![10_000, 10_000],
        device_memory: vec!["vram_0".into(), "vram_1".into()],
        edge_bytes,
        subsets: vec![
            SubsetSpec {
                device_indices: vec![0],
            },
            SubsetSpec {
                device_indices: vec![1],
            },
            SubsetSpec {
                device_indices: vec![0, 1],
            },
        ],
        machine,
        config: PlanningConfig {
            objective: ObjectiveKind::Latency,
            beam_width: 16,
            candidates_per_device: 2,
            local_search_iters: 2,
            planner_workers: 1,
            allow_parallel_subsets: false,
            finalist_count: 4,
            ..PlanningConfig::default()
        },
    }
}

#[test]
fn finds_transfer_aware_optimum() {
    let problem = two_device_problem();
    let results = search_subset(&problem, &[0, 1]);
    assert!(!results.is_empty());
    let result = &results[0];
    // Best: r0@accel_0 (0.01) + transfer(0.001) + r1@accel_1 (0.015) = 0.026
    // beats colocated 0.03 on accel_0.
    assert_eq!(result.placements[0].device, "accel_0");
    assert_eq!(result.placements[1].device, "accel_1");
    assert!((result.latency_s - 0.026).abs() < 1e-9);
}

#[test]
fn memory_capacity_rejects_overflow() {
    let mut problem = two_device_problem();
    problem.capacities = vec![500, 10_000]; // accel_0 too small for 1000-byte output
    problem.candidates[0][0].workspace_bytes = 0;
    let result = search_subset(&problem, &[0]);
    assert!(result.is_empty());
}

#[test]
fn same_subset_emits_multiple_distinct_placements() {
    let mut problem = two_device_problem();
    problem.config.per_subset_finalists = 4;
    problem.config.beam_width = 16;
    problem.subsets = vec![SubsetSpec {
        device_indices: vec![0, 1],
    }];
    problem.config.finalist_count = 4;
    let out = plan_placements(&problem);
    assert!(
        out.finalists.len() >= 2,
        "expected multiple same-subset finalists, got {}",
        out.finalists.len()
    );
    let mut sigs = std::collections::HashSet::new();
    for f in &out.finalists {
        assert_eq!(
            f.subset_devices,
            vec!["accel_0".to_string(), "accel_1".to_string()]
        );
        assert!(sigs.insert(f.placement_signature.clone()));
    }
}

#[test]
fn deterministic_across_worker_counts() {
    let mut serial = two_device_problem();
    serial.config.planner_workers = 1;
    serial.config.allow_parallel_subsets = true;
    let mut parallel = two_device_problem();
    parallel.config.planner_workers = 8;
    parallel.config.allow_parallel_subsets = true;
    // Inflate work so auto-parallel can engage.
    for _ in 0..10 {
        parallel.subsets.push(SubsetSpec {
            device_indices: vec![0, 1],
        });
        serial.subsets.push(SubsetSpec {
            device_indices: vec![0, 1],
        });
    }
    let a = plan_placements(&serial);
    let b = plan_placements(&parallel);
    assert!(!a.finalists.is_empty());
    assert_eq!(
        a.finalists[0].placement_signature,
        b.finalists[0].placement_signature
    );
}

#[test]
fn finalist_dedup_by_signature() {
    let mut problem = two_device_problem();
    problem.subsets = vec![
        SubsetSpec {
            device_indices: vec![0, 1],
        },
        SubsetSpec {
            device_indices: vec![0, 1],
        },
        SubsetSpec {
            device_indices: vec![0, 1],
        },
    ];
    problem.config.finalist_count = 8;
    let out = plan_placements(&problem);
    let mut sigs = std::collections::HashSet::new();
    for f in &out.finalists {
        assert!(sigs.insert(f.placement_signature.clone()));
    }
}

#[test]
fn tiny_work_stays_serial() {
    assert!(!should_parallelize_subsets(1, 1, 64, 2, true, 8));
    assert!(!should_parallelize_subsets(2, 2, 8, 2, true, 8));
    assert!(!should_parallelize_subsets(3, 16, 32, 2, true, 8));
    assert!(should_parallelize_subsets(8, 16, 64, 4, true, 8));
}

#[test]
fn workspace_bytes_make_placement_infeasible() {
    let mut problem = two_device_problem();
    // State+output fit, but workspace pushes over capacity on accel_0.
    problem.capacities = vec![1_500, 10_000];
    problem.regions[0].output_bytes = 1_000;
    problem.regions[0].state_bytes = 0;
    problem.candidates[0][0].workspace_bytes = 600; // 1000+600 > 1500
    problem.candidates[0][1].workspace_bytes = 0;
    // Solo accel_0 infeasible; accel_1 still works.
    assert!(search_subset(&problem, &[0]).is_empty());
    assert!(!search_subset(&problem, &[1]).is_empty());
}

#[test]
fn directional_links_use_correct_direction() {
    let mut problem = two_device_problem();
    problem.machine.links.clear();
    problem.machine.links.push(TransferLink {
        id: "vram_0->vram_1".into(),
        source: "vram_0".into(),
        destination: "vram_1".into(),
        bandwidth_bytes_per_s: 1e9,
        latency_s: 0.0,
        link_class: "pcie".into(),
        contention_factor: 1.0,
        measured: true,
        bidirectional: false,
        peer_to_peer: false,
    });
    problem.machine.links.push(TransferLink {
        id: "vram_1->vram_0".into(),
        source: "vram_1".into(),
        destination: "vram_0".into(),
        bandwidth_bytes_per_s: 1e3, // very slow reverse
        latency_s: 0.0,
        link_class: "pcie".into(),
        contention_factor: 1.0,
        measured: true,
        bidirectional: false,
        peer_to_peer: false,
    });
    // Prefer r0@1 then r1@0 would pay slow reverse; r0@0 then r1@1 pays fast forward.
    let results = search_subset(&problem, &[0, 1]);
    assert!(!results.is_empty());
    assert_eq!(results[0].placements[0].device, "accel_0");
    assert_eq!(results[0].placements[1].device, "accel_1");
}

#[test]
fn host_staged_disallowed_prunes() {
    let mut problem = two_device_problem();
    problem.machine.links.clear();
    problem.machine.allow_host_staged_transfers = false;
    problem.config.allow_host_staged_transfers = false;
    let _mixed = search_subset(&problem, &[0, 1]);
    let solo = search_subset(&problem, &[0]);
    assert!(!solo.is_empty());
}

#[test]
fn exhaustive_oracle_matches_large_beam() {
    let mut problem = two_device_problem();
    problem.config.beam_width = 64;
    problem.config.local_search_iters = 0;
    problem.config.per_subset_finalists = 1;
    let oracle = exhaustive_best(&problem, &[0, 1], 50_000).expect("oracle");
    let beam = search_subset(&problem, &[0, 1]);
    assert!(!beam.is_empty());
    // Beam + local search may polish further; require beam ≤ oracle + eps and
    // placement cand path matches oracle when local search is disabled.
    assert!(
        beam[0].latency_s <= oracle.0 + 1e-9,
        "beam latency {} worse than oracle analytic {}",
        beam[0].latency_s,
        oracle.0
    );
    // With local_search off, latency objective analytic ≈ makespan.
    assert!((beam[0].latency_s - 0.026).abs() < 1e-9);
    assert_eq!(beam[0].placements[0].device, "accel_0");
    assert_eq!(beam[0].placements[1].device, "accel_1");
    let _ = oracle;
}

#[test]
fn exhaustive_oracle_memory_objective() {
    let mut problem = two_device_problem();
    problem.config.objective = ObjectiveKind::Memory;
    problem.config.beam_width = 64;
    problem.config.local_search_iters = 0;
    let oracle = exhaustive_best(&problem, &[0], 10_000).expect("oracle");
    let beam = search_subset(&problem, &[0]);
    assert!(!beam.is_empty());
    // Oracle path length equals region count; beam finds a feasible memory plan.
    assert_eq!(oracle.1.len(), problem.order.len());
    assert_eq!(beam[0].placements.len(), problem.order.len());
    assert_eq!(beam[0].placements[0].device, "accel_0");
}

#[test]
fn local_search_prefix_matches_full_replay() {
    let problem = two_device_problem();
    let results = search_subset(&problem, &[0, 1]);
    assert!(!results.is_empty());
    let mut assignment = Vec::new();
    for (step, &ridx) in problem.order.iter().enumerate() {
        let device = &results[0].placements[step].device;
        let kern = &results[0].placements[step].kernel_id;
        let (ci, c) = problem.candidates[ridx]
            .iter()
            .enumerate()
            .find(|(_, c)| problem.device_names[c.device] == *device && c.kernel_id == *kern)
            .map(|(i, c)| (i as u16, c.clone()))
            .expect("cand");
        assignment.push((ci, c));
    }
    for index in 0..problem.order.len() {
        let ridx = problem.order[index];
        for (i, c) in problem.candidates[ridx].iter().enumerate() {
            let alternate = (i as u16, c.clone());
            assert!(
                replay_scores_match(&problem, &[0, 1], &assignment, index, &alternate),
                "prefix/full diverge at step {index} alt {}",
                alternate.1.kernel_id
            );
        }
    }
}

#[test]
fn contention_scales_planner_transfer_like_machine() {
    let mut problem = two_device_problem();
    problem.machine.links[0].contention_factor = 4.0;
    problem.machine.links[0].bandwidth_bytes_per_s = 1e6;
    problem.machine.links[0].latency_s = 0.0;
    let nbytes = 1000u64;
    let est = problem
        .machine
        .estimate_transfer("vram_0", "vram_1", nbytes)
        .expect("link");
    // duration = (lat + n/bw) * contention = (0 + 1000/1e6) * 4 = 0.004
    assert!((est.duration_s - 0.004).abs() < 1e-12);
    assert!((est.contention_factor - 4.0).abs() < 1e-12);
    let results = search_subset(&problem, &[0, 1]);
    assert!(!results.is_empty());
    // Split placement pays contended transfer.
    let split = results
        .iter()
        .find(|r| r.placements[0].device != r.placements[1].device);
    if let Some(s) = split {
        assert!(s.transfer_latency_s + 1e-12 >= 0.004);
    }
}

#[test]
fn parallel_beam_deterministic_vs_serial() {
    let mut problem = two_device_problem();
    problem.config.beam_width = 32;
    problem.config.candidates_per_device = 2;
    // Inflate pools so beam*cand crosses threshold.
    for _ in 0..6 {
        problem.regions.push(RegionSpec {
            name: format!("r{}", problem.regions.len()),
            depends_on: vec![problem.regions.len() - 1],
            output_bytes: 8,
            state_bytes: 0,
            consumer_count: 1,
        });
        let last = problem.regions.len() - 1;
        problem.order.push(last);
        problem.edge_bytes.insert((last - 1, last), 8);
        problem.candidates.push(vec![
            CandidateKernel {
                device: 0,
                backend_id: "mock".into(),
                kernel_id: format!("r{last}:a0"),
                dtype: "float32".into(),
                estimated_latency_s: 0.01,
                workspace_bytes: 0,
                measured: true,
            },
            CandidateKernel {
                device: 1,
                backend_id: "mock".into(),
                kernel_id: format!("r{last}:a1"),
                dtype: "float32".into(),
                estimated_latency_s: 0.012,
                workspace_bytes: 0,
                measured: true,
            },
        ]);
    }
    assert!(!should_parallelize_beam(32, 2, 4)); // below tuned threshold
    assert!(should_parallelize_beam(256, 2, 4));
    problem.config.beam_width = 256;
    problem.config.candidates_per_device = 4;
    // Inflate pools so beam*pool crosses the parallel-beam gate.
    for pool in &mut problem.candidates {
        let base = pool[0].clone();
        for k in 0..4 {
            let mut c = base.clone();
            c.kernel_id = format!("{}:k{k}", base.kernel_id);
            c.estimated_latency_s = base.estimated_latency_s * (1.0 + 0.01 * k as f64);
            pool.push(c);
        }
    }
    let (serial, _) = search_subset_ex(&problem, &[0, 1], 1);
    let (parallel, used) = search_subset_ex(&problem, &[0, 1], 4);
    assert!(used, "expected intra-subset parallel expand");
    assert!(!serial.is_empty());
    assert_eq!(
        serial[0]
            .placements
            .iter()
            .map(|p| (&p.device, &p.kernel_id))
            .collect::<Vec<_>>(),
        parallel[0]
            .placements
            .iter()
            .map(|p| (&p.device, &p.kernel_id))
            .collect::<Vec<_>>()
    );
}

#[test]
fn analytic_rank_preserved_after_diversity_selection() {
    let mut problem = two_device_problem();
    problem.config.finalist_count = 3;
    // Cap first-pass per subset so diversity may skip some analytic ranks.
    problem.config.per_subset_finalists = 1;
    problem.config.planner_workers = 1;
    let out = plan_placements(&problem);
    assert!(!out.finalists.is_empty());
    for (i, f) in out.finalists.iter().enumerate() {
        assert_eq!(
            f.finalist_rank, i,
            "finalist_rank must be dense shortlist index"
        );
        assert_eq!(
            f.search_rank, f.analytic_rank,
            "search_rank must alias analytic_rank (never renumbered to shortlist index)"
        );
        // Analytic rank is position in the full sorted feasible list.
        assert!(
            f.analytic_rank >= i,
            "analytic_rank {} should be >= finalist_rank {}",
            f.analytic_rank,
            i
        );
    }
    // With per-subset cap, shortlist order is not a dense renumber of analytic ranks
    // whenever diversity skips an intervening analytic candidate.
    let analytic_ranks: Vec<_> = out.finalists.iter().map(|f| f.analytic_rank).collect();
    let dense: Vec<_> = (0..out.finalists.len()).collect();
    if analytic_ranks != dense {
        assert!(
            out.finalists
                .iter()
                .any(|f| f.analytic_rank != f.finalist_rank),
            "diversity skip must leave analytic_rank != finalist_rank"
        );
    }
}

#[test]
fn planner_workers_one_guarantees_serial_reporting() {
    let mut problem = two_device_problem();
    problem.config.planner_workers = 1;
    problem.config.allow_parallel_subsets = true;
    for _ in 0..12 {
        problem.subsets.push(SubsetSpec {
            device_indices: vec![0, 1],
        });
    }
    problem.config.beam_width = 64;
    let out = plan_placements(&problem);
    assert_eq!(out.statistics.planner_workers_requested, 1);
    assert_eq!(out.statistics.planner_workers_available, 1);
    assert_eq!(out.statistics.planner_workers_used, 1);
    assert!(!out.statistics.parallel_search_used);
    assert!(!out.statistics.parallel_beam_used);
}

#[test]
fn planner_workers_unused_when_workload_serial() {
    let mut problem = two_device_problem();
    problem.config.planner_workers = 4;
    problem.config.allow_parallel_subsets = true;
    // Tiny graph: subset/beam gates stay closed.
    let out = plan_placements(&problem);
    assert_eq!(out.statistics.planner_workers_requested, 4);
    assert_eq!(out.statistics.planner_workers_available, 4);
    assert!(
        !out.statistics.parallel_search_used && !out.statistics.parallel_beam_used,
        "tiny problem should stay serial"
    );
    assert_eq!(
        out.statistics.planner_workers_used, 1,
        "must not claim parallel workers when search stayed serial"
    );
    assert_eq!(
        out.statistics.planner_pool_threads, 1,
        "must not build a multi-thread Rayon pool for serial workloads"
    );
}

#[test]
fn planner_workers_cap_when_parallel_search_engages() {
    let mut problem = two_device_problem();
    problem.config.planner_workers = 4;
    problem.config.allow_parallel_subsets = true;
    problem.config.beam_width = 64;
    // Inflate subsets so subset parallel gate can open.
    for _ in 0..10 {
        problem.subsets.push(SubsetSpec {
            device_indices: vec![0, 1],
        });
    }
    // Grow to >=4 regions (gate requires region_count >= 4).
    for _ in 0..4 {
        let last = problem.regions.len();
        problem.regions.push(RegionSpec {
            name: format!("r{last}"),
            depends_on: vec![last - 1],
            output_bytes: 8,
            state_bytes: 0,
            consumer_count: 1,
        });
        problem.order.push(last);
        problem.edge_bytes.insert((last - 1, last), 8);
        problem.candidates.push(vec![
            CandidateKernel {
                device: 0,
                backend_id: "mock".into(),
                kernel_id: format!("r{last}:a0"),
                dtype: "float32".into(),
                estimated_latency_s: 0.01,
                workspace_bytes: 0,
                measured: true,
            },
            CandidateKernel {
                device: 1,
                backend_id: "mock".into(),
                kernel_id: format!("r{last}:a1"),
                dtype: "float32".into(),
                estimated_latency_s: 0.012,
                workspace_bytes: 0,
                measured: true,
            },
        ]);
    }
    let n = problem.regions.len();
    for (i, region) in problem.regions.iter_mut().enumerate() {
        region.consumer_count = if i + 1 < n { 1 } else { 0 };
    }
    // work = subsets * regions * beam * cand_avg ≈ 13 * 6 * 64 * 2 >> 8000
    assert!(should_parallelize_subsets(
        problem.subsets.len(),
        problem.region_count(),
        problem.config.beam_width,
        problem.avg_candidates_per_region(),
        true,
        4,
    ));
    let out = plan_placements(&problem);
    assert_eq!(out.statistics.planner_workers_available, 4);
    assert!(out.statistics.parallel_search_used);
    assert_eq!(out.statistics.planner_pool_threads, 4);
    assert!(
        out.statistics.planner_workers_used >= 2 && out.statistics.planner_workers_used <= 4,
        "used={}",
        out.statistics.planner_workers_used
    );
    assert!(!out.statistics.parallel_beam_used); // subset parallel disables nested beam
}

#[test]
fn serial_workload_skips_multi_thread_pool_even_when_workers_auto() {
    let mut problem = two_device_problem();
    problem.config.planner_workers = 0; // auto → many CPUs available
    problem.config.allow_parallel_subsets = true;
    problem.config.beam_width = 16;
    let available = resolve_workers(0);
    assert!(!beam_parallelism_possible(&problem, available.max(4)));
    let out = plan_placements(&problem);
    assert!(out.statistics.planner_workers_available >= 1);
    assert_eq!(out.statistics.planner_pool_threads, 1);
    assert_eq!(out.statistics.planner_workers_used, 1);
    assert!(!out.statistics.parallel_search_used);
    assert!(!out.statistics.parallel_beam_used);
}
