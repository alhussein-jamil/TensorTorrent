//! Property tests for schedule validation and serialization.

use proptest::prelude::*;
use sc_ir::{
    validate_schedule, validate_tensor_sizes, ExecutableSchedule, Instruction, InstructionId,
    MemoryTier, Opcode, RegionId, ResourceId, TensorId,
};

fn opcode_strategy() -> impl Strategy<Value = Opcode> {
    prop_oneof![
        Just(Opcode::Compute),
        Just(Opcode::Load),
        Just(Opcode::Transfer),
        Just(Opcode::Release),
        Just(Opcode::RecordEvent),
        Just(Opcode::WaitEvent),
    ]
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn random_dag_roundtrip(n in 1usize..12) {
        let mut instructions = Vec::new();
        for i in 0..n {
            let deps = if i == 0 {
                vec![]
            } else {
                // Always depend on previous => acyclic.
                vec![InstructionId::new(format!("n{}", i - 1))]
            };
            instructions.push(Instruction {
                opcode: Opcode::Compute,
                name: InstructionId::new(format!("n{i}")),
                resource: ResourceId::new("cpu"),
                depends_on: deps,
                inputs: vec![TensorId::new("x")],
                outputs: vec![TensorId::new(format!("y{i}"))],
                nbytes: 8,
                memory_tier: MemoryTier::SystemRam,
                predicted_duration_s: 0.0,
                executable_ref: Some(RegionId::new(format!("r{i}"))),
                source: None,
                destination: None,
                backend_id: None,
                transfer_backend: None,
                sync_required: false,
                stream_id: Some(sc_ir::StreamId::new("cpu::compute")),
                copy_engine_id: None,
                link_id: None,
                io_queue_id: None,
                attributes: indexmap::IndexMap::new(),
            });
        }
        let schedule = ExecutableSchedule::new("g", "fp", instructions, vec![]);
        assert!(validate_schedule(&schedule).ok());
        let bytes = schedule.to_json_bytes().unwrap();
        let back = ExecutableSchedule::from_json_bytes(&bytes).unwrap();
        assert_eq!(schedule, back);
    }

    #[test]
    fn cycle_always_rejected(a in "[a-z]{1,4}", b in "[a-z]{1,4}") {
        prop_assume!(a != b);
        let ia = Instruction {
            opcode: Opcode::Compute,
            name: InstructionId::new(&a),
            resource: ResourceId::new("cpu"),
            depends_on: vec![InstructionId::new(&b)],
            inputs: vec![TensorId::new("x")],
            outputs: vec![TensorId::new("y")],
            nbytes: 1,
            memory_tier: MemoryTier::SystemRam,
            predicted_duration_s: 0.0,
            executable_ref: Some(RegionId::new("r")),
            source: None,
            destination: None,
            backend_id: None,
            transfer_backend: None,
            sync_required: false,
            stream_id: None,
            copy_engine_id: None,
            link_id: None,
            io_queue_id: None,
            attributes: indexmap::IndexMap::new(),
        };
        let ib = Instruction {
            name: InstructionId::new(&b),
            depends_on: vec![InstructionId::new(&a)],
            ..ia.clone()
        };
        let schedule = ExecutableSchedule::new("g", "fp", vec![ia, ib], vec![]);
        assert!(!validate_schedule(&schedule).ok());
    }

    #[test]
    fn multi_tensor_requires_sizes(_op in opcode_strategy()) {
        let inst = Instruction {
            opcode: Opcode::Compute,
            name: InstructionId::new("c"),
            resource: ResourceId::new("cpu"),
            depends_on: vec![],
            inputs: vec![TensorId::new("a"), TensorId::new("b")],
            outputs: vec![TensorId::new("c")],
            nbytes: 100,
            memory_tier: MemoryTier::SystemRam,
            predicted_duration_s: 0.0,
            executable_ref: Some(RegionId::new("r")),
            source: None,
            destination: None,
            backend_id: None,
            transfer_backend: None,
            sync_required: false,
            stream_id: None,
            copy_engine_id: None,
            link_id: None,
            io_queue_id: None,
            attributes: indexmap::IndexMap::new(),
        };
        let schedule = ExecutableSchedule::new("g", "fp", vec![inst], vec![]);
        assert!(validate_schedule(&schedule).ok());
        assert!(!validate_tensor_sizes(&schedule).ok());
    }
}
