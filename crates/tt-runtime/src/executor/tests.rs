use super::capacity::wait_for_resource;
use super::*;
use indexmap::IndexMap;
use parking_lot::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tt_ir::{Instruction, InstructionId, MemoryTier, RegionId};

#[test]
fn empty_schedule() {
    let schedule = ExecutableSchedule::new("g", "fp", vec![], vec![]);
    let report = execute_schedule(&schedule, &ExecuteOptions::default(), None, None).unwrap();
    assert_eq!(report.events.len(), 0);
}

#[test]
fn region_cb_wave_batches_independent_computes() {
    let mk = |name: &str, region: &str, out: &str| Instruction {
        opcode: Opcode::Compute,
        name: InstructionId::new(name),
        resource: ResourceId::new("cpu"),
        depends_on: vec![],
        inputs: vec![TensorId::new("x")],
        outputs: vec![TensorId::new(out)],
        nbytes: 8,
        memory_tier: MemoryTier::SystemRam,
        predicted_duration_s: 0.0,
        executable_ref: Some(RegionId::new(region)),
        source: None,
        destination: None,
        backend_id: None,
        transfer_backend: None,
        sync_required: false,
        stream_id: None,
        copy_engine_id: None,
        link_id: None,
        io_queue_id: None,
        attributes: IndexMap::new(),
    };
    let schedule = ExecutableSchedule::new(
        "g",
        "fp",
        vec![mk("a", "ra", "y"), mk("b", "rb", "z")],
        vec![],
    );
    assert!(max_ready_width(&schedule) >= 2);
    let calls = Arc::new(Mutex::new(Vec::<usize>::new()));
    let calls_c = Arc::clone(&calls);
    let cb: RegionCallback = Arc::new(move |invs| {
        calls_c.lock().push(invs.len());
        Ok(())
    });
    let report = execute_schedule(&schedule, &ExecuteOptions::default(), Some(cb), None).unwrap();
    assert_eq!(report.events.len(), 2);
    let sizes = calls.lock().clone();
    assert_eq!(sizes, vec![2], "both ready Computes in one region_cb wave");
    for ev in &report.events {
        assert!(
            ev.notes.contains("region_callback_batch:2"),
            "notes={}",
            ev.notes
        );
    }
}

#[test]
fn handle_release_wave_batches_ready_releases() {
    let mk_compute = |name: &str, region: &str, out: &str| Instruction {
        opcode: Opcode::Compute,
        name: InstructionId::new(name),
        resource: ResourceId::new("cpu"),
        depends_on: vec![],
        inputs: vec![TensorId::new("x")],
        outputs: vec![TensorId::new(out)],
        nbytes: 8,
        memory_tier: MemoryTier::SystemRam,
        predicted_duration_s: 0.0,
        executable_ref: Some(RegionId::new(region)),
        source: None,
        destination: None,
        backend_id: None,
        transfer_backend: None,
        sync_required: false,
        stream_id: None,
        copy_engine_id: None,
        link_id: None,
        io_queue_id: None,
        attributes: IndexMap::new(),
    };
    let mk_release = |name: &str, dep: &str, tid: &str| Instruction {
        opcode: Opcode::Release,
        name: InstructionId::new(name),
        resource: ResourceId::new("cpu"),
        depends_on: vec![InstructionId::new(dep)],
        inputs: vec![TensorId::new(tid)],
        outputs: vec![],
        nbytes: 8,
        memory_tier: MemoryTier::SystemRam,
        predicted_duration_s: 0.0,
        executable_ref: None,
        source: None,
        destination: None,
        backend_id: None,
        transfer_backend: None,
        sync_required: false,
        stream_id: None,
        copy_engine_id: None,
        link_id: None,
        io_queue_id: None,
        attributes: IndexMap::new(),
    };
    let schedule = ExecutableSchedule::new(
        "g",
        "fp",
        vec![
            mk_compute("c0", "r0", "a0"),
            mk_compute("c1", "r1", "a1"),
            mk_release("rel0", "c0", "a0"),
            mk_release("rel1", "c1", "a1"),
        ],
        vec![],
    );
    let ctx = NativeExecutionContext::new();
    // Seed residency so Release can final-drop.
    for tid in ["a0", "a1"] {
        let id = ctx.next_alloc_id();
        ctx.residency()
            .put(
                TensorId::new(tid),
                ResourceId::new("cpu"),
                id,
                TensorMetadata {
                    nbytes: 8,
                    ..Default::default()
                },
                None,
            )
            .unwrap();
    }
    let calls = Arc::new(Mutex::new(0usize));
    let tensors = Arc::new(Mutex::new(0usize));
    let calls_c = Arc::clone(&calls);
    let tensors_c = Arc::clone(&tensors);
    let href: HandleReleaseCallback = Arc::new(move |pairs| {
        *calls_c.lock() += 1;
        *tensors_c.lock() += pairs.len();
        Ok(())
    });
    let region: RegionCallback = Arc::new(|_invs| Ok(()));
    let opts = ExecuteOptions {
        handle_release: Some(href),
        ..Default::default()
    };
    let report = execute_schedule_with_context(&schedule, &opts, Some(region), None, ctx).unwrap();
    assert_eq!(report.events.len(), 4);
    assert_eq!(*tensors.lock(), 2, "both tensors released");
    assert_eq!(
        *calls.lock(),
        1,
        "ready Releases share one handle_release callback"
    );
}

#[test]
fn branching_dag_dry_run() {
    let a = Instruction {
        opcode: Opcode::Compute,
        name: InstructionId::new("a"),
        resource: ResourceId::new("cpu"),
        depends_on: vec![],
        inputs: vec![TensorId::new("x")],
        outputs: vec![TensorId::new("y")],
        nbytes: 8,
        memory_tier: MemoryTier::SystemRam,
        predicted_duration_s: 0.0,
        executable_ref: Some(RegionId::new("ra")),
        source: None,
        destination: None,
        backend_id: None,
        transfer_backend: None,
        sync_required: false,
        stream_id: None,
        copy_engine_id: None,
        link_id: None,
        io_queue_id: None,
        attributes: IndexMap::new(),
    };
    let b = Instruction {
        name: InstructionId::new("b"),
        depends_on: vec![InstructionId::new("a")],
        outputs: vec![TensorId::new("z")],
        executable_ref: Some(RegionId::new("rb")),
        ..a.clone()
    };
    let c = Instruction {
        name: InstructionId::new("c"),
        depends_on: vec![InstructionId::new("a")],
        outputs: vec![TensorId::new("w")],
        executable_ref: Some(RegionId::new("rc")),
        ..a.clone()
    };
    let join = Instruction {
        name: InstructionId::new("join"),
        depends_on: vec![InstructionId::new("b"), InstructionId::new("c")],
        inputs: vec![TensorId::new("z"), TensorId::new("w")],
        outputs: vec![TensorId::new("out")],
        executable_ref: Some(RegionId::new("rj")),
        attributes: {
            let mut m = IndexMap::new();
            m.insert(
                "tensor_nbytes".into(),
                tt_ir::AttrValue::IntMap(
                    [("z".into(), 8i64), ("w".into(), 8), ("out".into(), 8)]
                        .into_iter()
                        .collect(),
                ),
            );
            m
        },
        ..a.clone()
    };
    let schedule = ExecutableSchedule::new("g", "fp", vec![a, b, c, join], vec![]);
    let opts = ExecuteOptions {
        dry_run_compute: true,
        ..Default::default()
    };
    let report = execute_schedule(&schedule, &opts, None, None).unwrap();
    assert_eq!(report.events.len(), 4);
}

#[test]
fn native_launch_skips_region_callback() {
    use tt_ir::AttrValue;
    let mut attrs = IndexMap::new();
    attrs.insert("native_launch".into(), AttrValue::Bool(true));
    let a = Instruction {
        opcode: Opcode::Compute,
        name: InstructionId::new("a"),
        resource: ResourceId::new("mock_accel0"),
        depends_on: vec![],
        inputs: vec![TensorId::new("x")],
        outputs: vec![TensorId::new("y")],
        nbytes: 8,
        memory_tier: MemoryTier::SystemRam,
        predicted_duration_s: 0.0,
        executable_ref: Some(RegionId::new("ra")),
        source: None,
        destination: None,
        backend_id: None,
        transfer_backend: None,
        sync_required: false,
        stream_id: None,
        copy_engine_id: None,
        link_id: None,
        io_queue_id: None,
        attributes: attrs,
    };
    let schedule = ExecutableSchedule::new("g", "fp", vec![a], vec![]);
    let called = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&called);
    let cb: RegionCallback = Arc::new(move |_| {
        flag.store(true, Ordering::SeqCst);
        Ok(())
    });
    let report = execute_schedule(&schedule, &ExecuteOptions::default(), Some(cb), None).unwrap();
    assert_eq!(report.events.len(), 1);
    assert!(
        !called.load(Ordering::SeqCst),
        "native_launch must not invoke region callback"
    );
    assert!(report.events[0].simulated);
}

#[test]
fn rejects_cycle() {
    let a = Instruction {
        opcode: Opcode::Compute,
        name: InstructionId::new("a"),
        resource: ResourceId::new("cpu"),
        depends_on: vec![InstructionId::new("b")],
        inputs: vec![TensorId::new("x")],
        outputs: vec![TensorId::new("y")],
        nbytes: 8,
        memory_tier: MemoryTier::SystemRam,
        predicted_duration_s: 0.0,
        executable_ref: Some(RegionId::new("ra")),
        source: None,
        destination: None,
        backend_id: None,
        transfer_backend: None,
        sync_required: false,
        stream_id: None,
        copy_engine_id: None,
        link_id: None,
        io_queue_id: None,
        attributes: IndexMap::new(),
    };
    let b = Instruction {
        name: InstructionId::new("b"),
        depends_on: vec![InstructionId::new("a")],
        executable_ref: Some(RegionId::new("rb")),
        ..a.clone()
    };
    let schedule = ExecutableSchedule::new("g", "fp", vec![a, b], vec![]);
    assert!(execute_schedule(&schedule, &ExecuteOptions::default(), None, None).is_err());
}

#[test]
fn transfer_without_source_copy_fails() {
    let xfer = Instruction {
        opcode: Opcode::Transfer,
        name: InstructionId::new("t0"),
        resource: ResourceId::new("cpu"),
        depends_on: vec![],
        inputs: vec![TensorId::new("w")],
        outputs: vec![TensorId::new("w")],
        nbytes: 64,
        memory_tier: MemoryTier::SystemRam,
        predicted_duration_s: 0.0,
        executable_ref: None,
        source: Some(ResourceId::new("cpu")),
        destination: Some(ResourceId::new("mock0")),
        backend_id: None,
        transfer_backend: None,
        sync_required: false,
        stream_id: Some(tt_ir::StreamId::new("cpu::copy0")),
        copy_engine_id: Some("cpu::copy0".into()),
        link_id: Some("cpu->mock0".into()),
        io_queue_id: None,
        attributes: IndexMap::new(),
    };
    let schedule = ExecutableSchedule::new("g", "fp", vec![xfer], vec![]);
    let opts = ExecuteOptions {
        dry_run_compute: false,
        ..Default::default()
    };
    let err = execute_schedule(&schedule, &opts, None, None).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("source copy missing") || msg.contains("missing or stale"),
        "unexpected error: {msg}"
    );
}

#[test]
fn wait_event_without_record_in_table_fails() {
    let wait = Instruction {
        opcode: Opcode::WaitEvent,
        name: InstructionId::new("w0"),
        resource: ResourceId::new("cpu"),
        depends_on: vec![],
        inputs: vec![],
        outputs: vec![],
        nbytes: 0,
        memory_tier: MemoryTier::SystemRam,
        predicted_duration_s: 0.0,
        executable_ref: None,
        source: None,
        destination: None,
        backend_id: None,
        transfer_backend: None,
        sync_required: false,
        stream_id: Some(tt_ir::StreamId::new("cpu::compute0")),
        copy_engine_id: None,
        link_id: None,
        io_queue_id: None,
        attributes: {
            let mut m = IndexMap::new();
            m.insert(
                "waits_for".into(),
                tt_ir::AttrValue::String("never_recorded".into()),
            );
            m
        },
    };
    let ctx = NativeExecutionContext::new();
    let err = run_instruction(&wait, &ctx, None, false, &ExecuteOptions::default()).unwrap_err();
    assert!(
        err.to_string().contains("never recorded"),
        "unexpected: {err}"
    );
}

#[test]
fn shared_context_survives_region_path() {
    let load = Instruction {
        opcode: Opcode::Load,
        name: InstructionId::new("l0"),
        resource: ResourceId::new("cpu"),
        depends_on: vec![],
        inputs: vec![],
        outputs: vec![TensorId::new("p")],
        nbytes: 16,
        memory_tier: MemoryTier::SystemRam,
        predicted_duration_s: 0.0,
        executable_ref: None,
        source: Some(ResourceId::new("disk")),
        destination: Some(ResourceId::new("cpu")),
        backend_id: None,
        transfer_backend: None,
        sync_required: false,
        stream_id: Some(tt_ir::StreamId::new("cpu::io0")),
        copy_engine_id: None,
        link_id: None,
        io_queue_id: None,
        attributes: IndexMap::new(),
    };
    let compute = Instruction {
        opcode: Opcode::Compute,
        name: InstructionId::new("c0"),
        resource: ResourceId::new("cpu"),
        depends_on: vec![InstructionId::new("l0")],
        inputs: vec![TensorId::new("p"), TensorId::new("x")],
        outputs: vec![TensorId::new("y")],
        nbytes: 16,
        memory_tier: MemoryTier::SystemRam,
        predicted_duration_s: 0.0,
        executable_ref: Some(RegionId::new("r0")),
        source: None,
        destination: None,
        backend_id: None,
        transfer_backend: None,
        sync_required: false,
        stream_id: Some(tt_ir::StreamId::new("cpu::compute0")),
        copy_engine_id: None,
        link_id: None,
        io_queue_id: None,
        attributes: IndexMap::new(),
    };
    let schedule = ExecutableSchedule::new("g", "fp", vec![load, compute], vec![]);
    let ctx = NativeExecutionContext::new();
    // Prematerialize like Python would.
    let store = ctx.residency();
    store
        .put(
            TensorId::new("p"),
            ResourceId::new("cpu"),
            ctx.next_alloc_id(),
            TensorMetadata {
                nbytes: 16,
                ..Default::default()
            },
            None,
        )
        .unwrap();
    store
        .put(
            TensorId::new("x"),
            ResourceId::new("cpu"),
            ctx.next_alloc_id(),
            TensorMetadata {
                nbytes: 16,
                ..Default::default()
            },
            None,
        )
        .unwrap();
    let called = Arc::new(AtomicBool::new(false));
    let called2 = Arc::clone(&called);
    let cb: RegionCallback = Arc::new(move |_invs| {
        called2.store(true, Ordering::Release);
        Ok(())
    });
    let report = execute_schedule_with_context(
        &schedule,
        &ExecuteOptions::default(),
        Some(cb),
        None,
        Arc::clone(&ctx),
    )
    .unwrap();
    assert!(called.load(Ordering::Acquire));
    assert_eq!(report.events.len(), 2);
    assert!(store
        .get(&TensorId::new("y"), &ResourceId::new("cpu"))
        .is_ok());
}

#[test]
fn activation_spill_without_io_handler_fails_closed() {
    let mut attrs = IndexMap::new();
    attrs.insert(
        "kind".into(),
        tt_ir::AttrValue::String("activation_spill".into()),
    );
    let spill = Instruction {
        opcode: Opcode::Evict,
        name: InstructionId::new("spill0"),
        resource: ResourceId::new("cpu"),
        depends_on: vec![],
        inputs: vec![TensorId::new("act")],
        outputs: vec![],
        nbytes: 64,
        memory_tier: MemoryTier::SystemRam,
        predicted_duration_s: 0.0,
        executable_ref: None,
        source: None,
        destination: None,
        backend_id: None,
        transfer_backend: None,
        sync_required: false,
        stream_id: None,
        copy_engine_id: None,
        link_id: None,
        io_queue_id: None,
        attributes: attrs,
    };
    let ctx = NativeExecutionContext::new();
    ctx.residency()
        .put(
            TensorId::new("act"),
            ResourceId::new("cpu"),
            ctx.next_alloc_id(),
            TensorMetadata {
                nbytes: 64,
                ..Default::default()
            },
            None,
        )
        .unwrap();
    let err = run_instruction(&spill, &ctx, None, false, &ExecuteOptions::default()).unwrap_err();
    assert!(
        err.to_string().contains("dematerialize") || err.to_string().contains("activation_spill"),
        "unexpected: {err}"
    );
    assert!(
        ctx.residency()
            .get(&TensorId::new("act"), &ResourceId::new("cpu"))
            .is_ok(),
        "spill must not drop RAM when body missing"
    );
}

#[test]
fn wait_for_resource_stalls_with_diagnosable_error() {
    let ctx = NativeExecutionContext::new();
    ctx.set_stall_timeout_secs(0.15);
    let started = std::time::Instant::now();
    let err = wait_for_resource(&ctx, "test capacity", || false).unwrap_err();
    assert!(matches!(*err, RuntimeError::Stalled { .. }), "got {err:?}");
    // Must fire promptly after the stall window, never hang.
    assert!(started.elapsed() < std::time::Duration::from_secs(5));
}

#[test]
fn wait_for_resource_resets_deadline_on_progress() {
    let ctx = NativeExecutionContext::new();
    ctx.set_stall_timeout_secs(0.2);
    let tries = std::sync::atomic::AtomicUsize::new(0);
    // Progress bumps keep the watchdog fed until acquisition succeeds.
    let ok = wait_for_resource(&ctx, "test capacity", || {
        let n = tries.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        ctx.bump_progress();
        n > 400
    });
    assert!(ok.is_ok());
}

#[test]
fn spill_session_dir_created_lazily_and_removed_on_drop() {
    let base = std::env::temp_dir().join(format!("tt_exec_spill_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    let session = {
        let ctx = NativeExecutionContext::new();
        ctx.set_spill_dir(base.clone());
        // Session dirs on tmpfs are refused unless the test escape hatch is set;
        // temp_dir in CI may be tmpfs, so allow it for this lifecycle test.
        std::env::set_var("TT_ALLOW_TMPFS_SPILL", "1");
        let session = ctx.spill_session_dir().unwrap();
        assert!(session.is_dir());
        assert!(session
            .file_name()
            .unwrap()
            .to_string_lossy()
            .starts_with("tt-spill-"));
        session
    };
    // Context dropped: the whole session (and any spill files) must be gone.
    assert!(!session.exists());
    let _ = std::fs::remove_dir_all(&base);
}
