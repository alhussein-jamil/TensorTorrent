//! Criterion benches for native schedule overhead.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use indexmap::IndexMap;
use streamcompiler_core::{
    ExecutableSchedule, Instruction, InstructionId, MemoryTier, Opcode, RegionId, ResourceId,
    TensorId,
};
use streamcompiler_runtime::{execute_schedule, ExecuteOptions};

fn make_linear(n: usize) -> ExecutableSchedule {
    let mut instructions = Vec::with_capacity(n);
    for i in 0..n {
        let deps = if i == 0 {
            vec![]
        } else {
            vec![InstructionId::new(format!("c{}", i - 1))]
        };
        instructions.push(Instruction {
            opcode: Opcode::Compute,
            name: InstructionId::new(format!("c{i}")),
            resource: ResourceId::new("cpu"),
            depends_on: deps,
            inputs: vec![TensorId::new("x")],
            outputs: vec![TensorId::new(format!("y{i}"))],
            nbytes: 64,
            memory_tier: MemoryTier::SystemRam,
            predicted_duration_s: 0.0,
            executable_ref: Some(RegionId::new(format!("r{i}"))),
            source: None,
            destination: None,
            backend_id: Some("cpu".into()),
            transfer_backend: None,
            sync_required: false,
            stream_id: None,
            copy_engine_id: None,
            link_id: None,
            attributes: IndexMap::new(),
        });
    }
    ExecutableSchedule::new("bench", "fp", instructions, vec![])
}

fn bench_empty(c: &mut Criterion) {
    let schedule = ExecutableSchedule::new("e", "f", vec![], vec![]);
    let opts = ExecuteOptions {
        dry_run_compute: true,
        ..Default::default()
    };
    c.bench_function("empty_schedule", |b| {
        b.iter(|| {
            execute_schedule(black_box(&schedule), &opts, None, None).unwrap();
        })
    });
}

fn bench_linear_64(c: &mut Criterion) {
    let schedule = make_linear(64);
    let opts = ExecuteOptions {
        dry_run_compute: true,
        ..Default::default()
    };
    c.bench_function("linear_64_dry_run", |b| {
        b.iter(|| {
            execute_schedule(black_box(&schedule), &opts, None, None).unwrap();
        })
    });
}

criterion_group!(benches, bench_empty, bench_linear_64);
criterion_main!(benches);
