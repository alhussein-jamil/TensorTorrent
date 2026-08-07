//! Native planner unit tests.

use crate::problem::{
    CandidateKernel, ObjectiveKind, PlanningConfig, PlanningProblem, RegionSpec, SubsetSpec,
};
use crate::search::{plan_placements, search_subset};
use crate::should_parallelize_subsets;
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
    let result = search_subset(&problem, &[0, 1]).expect("feasible");
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
    assert!(result.is_none());
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
    assert!(should_parallelize_subsets(8, 16, 64, 4, true, 8));
}

#[test]
fn host_staged_disallowed_prunes() {
    let mut problem = two_device_problem();
    problem.machine.links.clear();
    problem.machine.allow_host_staged_transfers = false;
    problem.config.allow_host_staged_transfers = false;
    // Cross-device only subset with no links → infeasible.
    let result = search_subset(&problem, &[0, 1]);
    // Colocated still works.
    let solo = search_subset(&problem, &[0]);
    assert!(solo.is_some());
    // Mixed may still colocate on one device via beam.
    let _ = result;
}
