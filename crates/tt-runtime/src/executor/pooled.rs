//! Dependency-counter schedule executor with resource queues and compute callbacks.

use crate::context::NativeExecutionContext;
use crate::error::{RuntimeError, RuntimeResult};
use crate::telemetry::InstructionTelemetry;
use crate::workers::WorkerPool;
use crossbeam_channel::{bounded, Receiver, Sender};
use parking_lot::Mutex;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Instant;
use tt_ir::{ExecutableSchedule, Opcode};

use super::capacity::{order_key, work_kind};
use super::instruction::{run_instruction, simulate_mock_compute};
use super::needs_python_io;
use super::waves::enqueue_ready;
use super::waves::{
    is_batched_handle_release, is_native_launch, run_parameter_load_wave, run_release_wave,
    take_ready_compute_wave, take_ready_parameter_load_wave, take_ready_release_wave,
};
use super::{
    Completion, ExecuteOptions, ExecuteReport, InstructionCallback, RegionCallback,
    RegionInvocation, WorkKind,
};
pub(crate) fn execute_schedule_pooled(
    schedule: &ExecutableSchedule,
    options: &ExecuteOptions,
    region_cb: Option<RegionCallback>,
    instruction_cb: Option<InstructionCallback>,
    ctx: Arc<NativeExecutionContext>,
    t0: Instant,
) -> RuntimeResult<ExecuteReport> {
    // Borrow view into the schedule instead of cloning every instruction up
    // front — jobs still clone once at dispatch (they must own their inputs
    // to cross thread boundaries), but this cuts the double-clone.
    let by_name: HashMap<&str, &tt_ir::Instruction> = schedule
        .instructions
        .iter()
        .map(|i| (i.name.as_str(), i))
        .collect();

    let mut remaining: HashMap<String, usize> = schedule
        .instructions
        .iter()
        .map(|i| (i.name.as_str().to_owned(), i.depends_on.len()))
        .collect();
    let mut dependents: HashMap<String, Vec<String>> = HashMap::new();
    for inst in &schedule.instructions {
        for dep in &inst.depends_on {
            dependents
                .entry(dep.as_str().to_owned())
                .or_default()
                .push(inst.name.as_str().to_owned());
        }
    }

    let mut ready: VecDeque<String> = remaining
        .iter()
        .filter_map(|(n, d)| if *d == 0 { Some(n.clone()) } else { None })
        .collect();

    // Ordered queues per accelerator-style resource.
    let mut resource_queues: HashMap<String, VecDeque<String>> = HashMap::new();
    let mut resource_busy: HashSet<String> = HashSet::new();

    let cancel = ctx.cancel_flag();
    // Track schedule-level completion names separately from RecordEvent table.
    let schedule_done: Arc<Mutex<HashSet<String>>> = Arc::new(Mutex::new(HashSet::new()));

    let cpu_pool = WorkerPool::try_new("cpu", options.cpu_workers, options.max_inflight)?;
    let io_pool = WorkerPool::try_new("io", options.io_workers, options.max_inflight)?;
    let xfer_pool = WorkerPool::try_new("xfer", options.transfer_workers, options.max_inflight)?;

    let (done_tx, done_rx): (Sender<Completion>, Receiver<Completion>) =
        bounded(options.max_inflight.max(8));

    let mut events = Vec::new();
    let mut inflight = 0usize;
    let mut bytes_read = 0u64;
    let mut bytes_transferred = 0u64;
    let mut simulated_ops = 0usize;
    let mut failure: Option<Box<RuntimeError>> = None;
    // Progress watchdog for the dispatch loop itself.
    let mut stall_clock = std::time::Instant::now();
    let total = schedule.instructions.len();
    let mut finished = 0usize;

    let origin = t0;

    while finished < total {
        if cancel.load(Ordering::Acquire) {
            failure = Some(Box::new(RuntimeError::Cancelled));
            break;
        }

        // Drain completions first.
        while let Ok(comp) = {
            let r = done_rx.try_recv();
            if r.is_ok() {
                ctx.bump_progress();
            }
            r
        } {
            inflight = inflight.saturating_sub(1);
            match comp.result {
                Ok(tel) => {
                    if tel.opcode == "Load" || tel.opcode == "Prefetch" {
                        bytes_read += tel.nbytes;
                    }
                    if tel.opcode == "Transfer" {
                        bytes_transferred += tel.nbytes;
                    }
                    if tel.simulated {
                        simulated_ops += 1;
                    }
                    // Free ordered stream/engine slot.
                    if let Some(ref key) = comp.order_key {
                        resource_busy.remove(key);
                        if let Some(q) = resource_queues.get_mut(key) {
                            if let Some(next) = q.pop_front() {
                                ready.push_back(next);
                            }
                        }
                    }
                    schedule_done.lock().insert(comp.name.clone());
                    // Unlock dependents.
                    if let Some(nexts) = dependents.get(&comp.name) {
                        for nxt in nexts {
                            if let Some(deg) = remaining.get_mut(nxt) {
                                *deg = deg.saturating_sub(1);
                                if *deg == 0 {
                                    enqueue_ready(
                                        nxt,
                                        &by_name,
                                        &mut ready,
                                        &mut resource_queues,
                                        &resource_busy,
                                    );
                                }
                            }
                        }
                    }
                    events.push(tel);
                    finished += 1;
                }
                Err(e) => {
                    failure = Some(e);
                    finished = total; // force exit
                    break;
                }
            }
        }

        if failure.is_some() {
            break;
        }

        // Launch ready work up to max_inflight.
        while inflight < options.max_inflight {
            // Wave-batch ready Releases / parameter_evicts: residency per-op, one GIL.
            // Do this before I/O launch so freed RAM admits the next Load promptly.
            if options.handle_release.is_some() {
                let release_names = take_ready_release_wave(&mut ready, &by_name);
                if !release_names.is_empty() {
                    let wave_insts: Vec<&tt_ir::Instruction> = release_names
                        .iter()
                        .filter_map(|n| by_name.get(n.as_str()).copied())
                        .collect();
                    match run_release_wave(&wave_insts, ctx.as_ref(), options, origin) {
                        Ok(teles) => {
                            for (tel, name) in teles.into_iter().zip(release_names.iter()) {
                                schedule_done.lock().insert(name.clone());
                                if let Some(nexts) = dependents.get(name) {
                                    for nxt in nexts {
                                        if let Some(deg) = remaining.get_mut(nxt) {
                                            *deg = deg.saturating_sub(1);
                                            if *deg == 0 {
                                                enqueue_ready(
                                                    nxt,
                                                    &by_name,
                                                    &mut ready,
                                                    &mut resource_queues,
                                                    &resource_busy,
                                                );
                                            }
                                        }
                                    }
                                }
                                events.push(tel);
                                finished += 1;
                            }
                        }
                        Err(e) => {
                            failure = Some(e);
                            finished = total;
                            break;
                        }
                    }
                    continue;
                }
            }

            // Wave-batch ready parameter Loads: one GIL for the whole wave.
            if options.parameter_load.is_some() {
                let load_names = take_ready_parameter_load_wave(&mut ready, &by_name);
                if !load_names.is_empty() {
                    let wave_insts: Vec<&tt_ir::Instruction> = load_names
                        .iter()
                        .filter_map(|n| by_name.get(n.as_str()).copied())
                        .collect();
                    match run_parameter_load_wave(&wave_insts, ctx.as_ref(), options, origin) {
                        Ok(teles) => {
                            for (tel, name) in teles.into_iter().zip(load_names.iter()) {
                                schedule_done.lock().insert(name.clone());
                                if let Some(nexts) = dependents.get(name) {
                                    for nxt in nexts {
                                        if let Some(deg) = remaining.get_mut(nxt) {
                                            *deg = deg.saturating_sub(1);
                                            if *deg == 0 {
                                                enqueue_ready(
                                                    nxt,
                                                    &by_name,
                                                    &mut ready,
                                                    &mut resource_queues,
                                                    &resource_busy,
                                                );
                                            }
                                        }
                                    }
                                }
                                events.push(tel);
                                finished += 1;
                            }
                        }
                        Err(e) => {
                            failure = Some(e);
                            finished = total;
                            break;
                        }
                    }
                    continue;
                }
            }

            // Prefer ready Prefetch/Transfer/Load before Compute waves so I/O
            // overlaps compute and input Transfers land before consumers.
            if let Some(name) = ready.iter().find(|n| {
                by_name
                    .get(n.as_str())
                    .map(|&i| {
                        matches!(
                            i.opcode,
                            Opcode::Prefetch | Opcode::Transfer | Opcode::Load | Opcode::Evict
                        ) && !is_batched_handle_release(i)
                    })
                    .unwrap_or(false)
            }) {
                let name = name.clone();
                ready.retain(|n| n != &name);
                let Some(&inst) = by_name.get(name.as_str()) else {
                    continue;
                };
                let ordered_key = order_key(inst);
                if let Some(ref key) = ordered_key {
                    if resource_busy.contains(key) {
                        resource_queues
                            .entry(key.clone())
                            .or_default()
                            .push_back(name);
                        continue;
                    }
                    resource_busy.insert(key.clone());
                }
                let kind = work_kind(inst.opcode);
                let inst = inst.clone();
                let name_c = name.clone();
                let ordered_key_c = ordered_key.clone();
                let done_tx = done_tx.clone();
                let ctx = Arc::clone(&ctx);
                let region_cb = region_cb.clone();
                let instruction_cb = instruction_cb.clone();
                let dry = options.dry_run_compute;
                let opts = options.clone();
                let job = move || {
                    let submitted = origin.elapsed().as_secs_f64();
                    let start = origin.elapsed().as_secs_f64();
                    let result = match (&region_cb, &instruction_cb) {
                        (Some(_rcb), Some(icb)) if needs_python_io(&inst, &opts) => {
                            match icb(inst.name.as_str()) {
                                Ok(r) => Ok((r.simulated, r.nbytes, r.notes)),
                                Err(cause) => Err(Box::new(RuntimeError::Instruction {
                                    instruction: inst.name.to_string(),
                                    opcode: inst.opcode.to_string(),
                                    region: inst.executable_ref.as_ref().map(|r| r.to_string()),
                                    tensor: inst.inputs.first().map(|t| t.to_string()),
                                    resource: Some(inst.resource.to_string()),
                                    cause,
                                })),
                            }
                        }
                        (_, Some(icb)) if region_cb.is_none() => match icb(inst.name.as_str()) {
                            Ok(r) => Ok((r.simulated, r.nbytes, r.notes)),
                            Err(cause) => Err(Box::new(RuntimeError::Instruction {
                                instruction: inst.name.to_string(),
                                opcode: inst.opcode.to_string(),
                                region: inst.executable_ref.as_ref().map(|r| r.to_string()),
                                tensor: inst.inputs.first().map(|t| t.to_string()),
                                resource: Some(inst.resource.to_string()),
                                cause,
                            })),
                        },
                        _ => run_instruction(&inst, &ctx, region_cb.as_ref(), dry, &opts).map(
                            |simulated| (simulated, inst.nbytes, String::from("native_data_plane")),
                        ),
                    };
                    let end = origin.elapsed().as_secs_f64();
                    let tel_result =
                        result.map(|(simulated, nbytes, notes)| InstructionTelemetry {
                            name: name_c.clone(),
                            opcode: inst.opcode.to_string(),
                            resource: inst.resource.to_string(),
                            submitted_s: submitted,
                            start_s: start,
                            end_s: end,
                            nbytes,
                            simulated,
                            notes,
                        });
                    let _ = done_tx.send(Completion {
                        name: name_c,
                        order_key: ordered_key_c,
                        result: tel_result,
                    });
                };
                let submitted = match kind {
                    WorkKind::Io => io_pool.submit(job),
                    WorkKind::Transfer => xfer_pool.submit(job),
                    WorkKind::Cpu | WorkKind::Inline => cpu_pool.submit(job),
                };
                if !submitted {
                    failure = Some(Box::new(RuntimeError::Other("worker queue closed".into())));
                    break;
                }
                inflight += 1;
                continue;
            }

            // Wave-batch ready Computes into one region_cb call (hybrid pooled path).
            if region_cb.is_some() {
                if let Some(wave) = take_ready_compute_wave(
                    &mut ready,
                    &by_name,
                    &mut resource_busy,
                    &mut resource_queues,
                ) {
                    let wave_insts: Vec<_> = wave
                        .iter()
                        .filter_map(|n| by_name.get(n.as_str()).copied().cloned())
                        .collect();
                    if wave_insts.is_empty() {
                        continue;
                    }
                    let (native_insts, py_insts): (Vec<_>, Vec<_>) =
                        wave_insts.into_iter().partition(is_native_launch);
                    for inst in native_insts {
                        let name_c = inst.name.as_str().to_owned();
                        let ordered_key_c = order_key(&inst);
                        let done_tx = done_tx.clone();
                        let ctx = Arc::clone(&ctx);
                        let opts = options.clone();
                        let job = move || {
                            let submitted = origin.elapsed().as_secs_f64();
                            let start = submitted;
                            let result =
                                run_instruction(&inst, &ctx, None, false, &opts).map(|simulated| {
                                    InstructionTelemetry {
                                        name: name_c.clone(),
                                        opcode: inst.opcode.to_string(),
                                        resource: inst.resource.to_string(),
                                        submitted_s: submitted,
                                        start_s: start,
                                        end_s: origin.elapsed().as_secs_f64(),
                                        nbytes: inst.nbytes,
                                        simulated,
                                        notes: "native_launch".into(),
                                    }
                                });
                            let _ = done_tx.send(Completion {
                                name: name_c,
                                order_key: ordered_key_c,
                                result,
                            });
                        };
                        if !cpu_pool.submit(job) {
                            failure =
                                Some(Box::new(RuntimeError::Other("worker queue closed".into())));
                            break;
                        }
                        inflight += 1;
                    }
                    let wave_insts = py_insts;
                    if wave_insts.is_empty() {
                        continue;
                    }
                    let order_keys: Vec<Option<String>> =
                        wave_insts.iter().map(order_key).collect();
                    let names: Vec<String> = wave_insts
                        .iter()
                        .map(|i| i.name.as_str().to_owned())
                        .collect();
                    let wave_len = names.len();
                    let done_tx = done_tx.clone();
                    let ctx = Arc::clone(&ctx);
                    let rcb = region_cb.clone().expect("region_cb checked");
                    let job = move || {
                        let submitted = origin.elapsed().as_secs_f64();
                        let start = origin.elapsed().as_secs_f64();
                        let invocations: Vec<RegionInvocation> = wave_insts
                            .iter()
                            .map(|inst| {
                                let region = inst
                                    .executable_ref
                                    .as_ref()
                                    .map(|r| r.as_str())
                                    .unwrap_or("");
                                RegionInvocation {
                                    region_id: region.to_owned(),
                                    inputs: inst.inputs.iter().map(|t| t.to_string()).collect(),
                                    outputs: inst.outputs.iter().map(|t| t.to_string()).collect(),
                                }
                            })
                            .collect();
                        let batch_note = if wave_insts.len() > 1 {
                            format!("region_callback_batch:{}", wave_insts.len())
                        } else {
                            "region_callback".into()
                        };
                        let batch_result = rcb(&invocations);
                        let end = origin.elapsed().as_secs_f64();
                        for (idx, inst) in wave_insts.iter().enumerate() {
                            let tel_result = match &batch_result {
                                Ok(()) => {
                                    let mut simulated = false;
                                    let sim_err = if inst.resource.as_str().contains("mock") {
                                        match simulate_mock_compute(inst, &ctx) {
                                            Ok(()) => {
                                                simulated = true;
                                                None
                                            }
                                            Err(e) => Some(e),
                                        }
                                    } else {
                                        None
                                    };
                                    if let Some(e) = sim_err {
                                        Err(e)
                                    } else {
                                        Ok(InstructionTelemetry {
                                            name: names[idx].clone(),
                                            opcode: inst.opcode.to_string(),
                                            resource: inst.resource.to_string(),
                                            submitted_s: submitted,
                                            start_s: start,
                                            end_s: end,
                                            nbytes: inst.nbytes,
                                            simulated,
                                            notes: batch_note.clone(),
                                        })
                                    }
                                }
                                Err(cause) => Err(Box::new(RuntimeError::Instruction {
                                    instruction: names[idx].clone(),
                                    opcode: "Compute".into(),
                                    region: inst.executable_ref.as_ref().map(|r| r.to_string()),
                                    tensor: None,
                                    resource: Some(inst.resource.to_string()),
                                    cause: cause.clone(),
                                })),
                            };
                            let _ = done_tx.send(Completion {
                                name: names[idx].clone(),
                                order_key: order_keys[idx].clone(),
                                result: tel_result,
                            });
                        }
                    };
                    if !cpu_pool.submit(job) {
                        failure = Some(Box::new(RuntimeError::Other("worker queue closed".into())));
                        break;
                    }
                    inflight += wave_len;
                    continue;
                }
            }

            let Some(name) = ready.pop_front() else {
                break;
            };
            let Some(&inst) = by_name.get(name.as_str()) else {
                continue;
            };

            // Ordered stream gate — key by stream_id / engine, not whole device.
            let ordered_key = order_key(inst);
            if let Some(ref key) = ordered_key {
                if resource_busy.contains(key) {
                    resource_queues
                        .entry(key.clone())
                        .or_default()
                        .push_back(name);
                    continue;
                }
                resource_busy.insert(key.clone());
            }

            let kind = work_kind(inst.opcode);
            let inst = inst.clone();
            let name_c = name.clone();
            let ordered_key_c = ordered_key.clone();
            let done_tx = done_tx.clone();
            let ctx = Arc::clone(&ctx);
            let region_cb = region_cb.clone();
            let instruction_cb = instruction_cb.clone();
            let dry = options.dry_run_compute;
            let opts = options.clone();

            let job = move || {
                let submitted = origin.elapsed().as_secs_f64();
                let start = origin.elapsed().as_secs_f64();
                let result = match (&region_cb, &instruction_cb) {
                    // Hybrid: non-Compute via instruction_cb / native; Computes handled above.
                    (Some(_rcb), Some(icb)) => {
                        if needs_python_io(&inst, &opts) {
                            match icb(inst.name.as_str()) {
                                Ok(r) => Ok((r.simulated, r.nbytes, r.notes)),
                                Err(cause) => Err(Box::new(RuntimeError::Instruction {
                                    instruction: inst.name.to_string(),
                                    opcode: inst.opcode.to_string(),
                                    region: inst.executable_ref.as_ref().map(|r| r.to_string()),
                                    tensor: inst.inputs.first().map(|t| t.to_string()),
                                    resource: Some(inst.resource.to_string()),
                                    cause,
                                })),
                            }
                        } else if inst.opcode == Opcode::Compute {
                            // Should have been wave-batched; fall back single for safety.
                            let region = inst
                                .executable_ref
                                .as_ref()
                                .map(|r| r.as_str())
                                .unwrap_or("");
                            let inv = RegionInvocation {
                                region_id: region.to_owned(),
                                inputs: inst.inputs.iter().map(|t| t.to_string()).collect(),
                                outputs: inst.outputs.iter().map(|t| t.to_string()).collect(),
                            };
                            match region_cb.as_ref().unwrap()(std::slice::from_ref(&inv)) {
                                Ok(()) => {
                                    if inst.resource.as_str().contains("mock") {
                                        match simulate_mock_compute(&inst, &ctx) {
                                            Ok(()) => Ok((
                                                true,
                                                inst.nbytes,
                                                String::from("region_callback"),
                                            )),
                                            Err(e) => Err(e),
                                        }
                                    } else {
                                        Ok((false, inst.nbytes, String::from("region_callback")))
                                    }
                                }
                                Err(cause) => Err(Box::new(RuntimeError::Instruction {
                                    instruction: inst.name.to_string(),
                                    opcode: inst.opcode.to_string(),
                                    region: Some(region.to_owned()),
                                    tensor: None,
                                    resource: Some(inst.resource.to_string()),
                                    cause,
                                })),
                            }
                        } else {
                            run_instruction(&inst, &ctx, None, dry, &opts).map(|simulated| {
                                (simulated, inst.nbytes, String::from("native_data_plane"))
                            })
                        }
                    }
                    // Full instruction-callback path (mock delays / non-region schedules):
                    // every opcode body runs in Python — never invent Rust residency.
                    (_, Some(icb)) => match icb(inst.name.as_str()) {
                        Ok(r) => Ok((r.simulated, r.nbytes, r.notes)),
                        Err(cause) => Err(Box::new(RuntimeError::Instruction {
                            instruction: inst.name.to_string(),
                            opcode: inst.opcode.to_string(),
                            region: inst.executable_ref.as_ref().map(|r| r.to_string()),
                            tensor: inst.inputs.first().map(|t| t.to_string()),
                            resource: Some(inst.resource.to_string()),
                            cause,
                        })),
                    },
                    (rcb, None) => run_instruction(&inst, &ctx, rcb.as_ref(), dry, &opts)
                        .map(|simulated| (simulated, inst.nbytes, String::new())),
                };
                let end = origin.elapsed().as_secs_f64();
                let tel_result = result.map(|(simulated, nbytes, notes)| InstructionTelemetry {
                    name: name_c.clone(),
                    opcode: inst.opcode.to_string(),
                    resource: inst.resource.to_string(),
                    submitted_s: submitted,
                    start_s: start,
                    end_s: end,
                    nbytes,
                    simulated,
                    notes,
                });
                let _ = done_tx.send(Completion {
                    name: name_c,
                    order_key: ordered_key_c,
                    result: tel_result,
                });
            };

            let submit_ok = match kind {
                WorkKind::Inline => {
                    job();
                    true
                }
                WorkKind::Cpu => cpu_pool.submit(job),
                WorkKind::Io => io_pool.submit(job),
                WorkKind::Transfer => xfer_pool.submit(job),
            };
            if !submit_ok {
                failure = Some(Box::new(RuntimeError::Other("worker queue closed".into())));
                break;
            }
            if !matches!(kind, WorkKind::Inline) {
                inflight += 1;
            } else {
                // Inline already completed via done_tx; will be drained next loop.
                inflight += 1;
            }
        }

        if (inflight == 0 && ready.is_empty() && finished < total)
            || (inflight > 0 && ready.is_empty())
        {
            // Wait for at least one completion without busy-wait, with a
            // progress watchdog: a worker that panicked or a completion lost
            // to a closed channel must surface as a diagnosable error, not a
            // silent hang.
            match done_rx.recv_timeout(std::time::Duration::from_millis(50)) {
                Ok(comp) => {
                    stall_clock = std::time::Instant::now();
                    let _ = done_tx.send(comp);
                }
                Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                    if let Some(limit) = ctx.stall_timeout() {
                        if stall_clock.elapsed() > limit {
                            failure = Some(Box::new(RuntimeError::Stalled {
                                what: format!(
                                    "schedule completions ({inflight} in flight, {finished}/{total} finished)"
                                ),
                                waited_s: stall_clock.elapsed().as_secs_f64(),
                            }));
                            break;
                        }
                    }
                }
                Err(_) => {
                    failure = Some(Box::new(RuntimeError::Other(
                        "completion channel closed".into(),
                    )));
                    break;
                }
            }
        } else {
            stall_clock = std::time::Instant::now();
        }
    }

    cpu_pool.join();
    io_pool.join();
    xfer_pool.join();

    if let Some(err) = failure {
        return Err(err);
    }

    Ok(ExecuteReport {
        wall_time_s: t0.elapsed().as_secs_f64(),
        events,
        peak_activation_bytes: ctx.peak_bytes(),
        allocation_peak_bytes: ctx.peak_bytes(),
        bytes_read,
        bytes_transferred,
        simulated_ops,
    })
}
