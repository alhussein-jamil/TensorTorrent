//! Dependency-counter schedule executor with resource queues and compute callbacks.

use crate::error::{RuntimeError, RuntimeResult};
use crate::telemetry::InstructionTelemetry;
use crate::workers::WorkerPool;
use crossbeam_channel::{bounded, Receiver, Sender};
use parking_lot::Mutex;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;
use streamcompiler_core::{assert_schedule_valid, ExecutableSchedule, Opcode};
use streamcompiler_core::{AllocationId, ResourceId, TensorId};
use streamcompiler_memory::{AllocationTable, ResidencyStore, TensorMetadata};

/// Callback invoked for Compute instructions. Acquired only around the call by Python bindings.
pub type RegionCallback =
    Arc<dyn Fn(&str, &[String], &[String]) -> Result<(), String> + Send + Sync>;

/// Full instruction dispatch callback: Rust schedules, Python/native backends execute.
/// Returns optional notes string.
pub type InstructionCallback =
    Arc<dyn Fn(&str) -> Result<InstructionCallbackResult, String> + Send + Sync>;

#[derive(Clone, Debug, Default)]
pub struct InstructionCallbackResult {
    pub nbytes: u64,
    pub simulated: bool,
    pub notes: String,
}

#[derive(Clone, Debug)]
pub struct ExecuteOptions {
    pub cpu_workers: usize,
    pub io_workers: usize,
    pub transfer_workers: usize,
    pub max_inflight: usize,
    /// When true, Compute is a timed no-op (native microbench / empty schedule).
    pub dry_run_compute: bool,
}

impl Default for ExecuteOptions {
    fn default() -> Self {
        Self {
            cpu_workers: 4,
            io_workers: 2,
            transfer_workers: 2,
            max_inflight: 64,
            dry_run_compute: false,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct ExecuteReport {
    pub wall_time_s: f64,
    pub events: Vec<InstructionTelemetry>,
    pub peak_activation_bytes: u64,
    pub allocation_peak_bytes: u64,
    pub bytes_read: u64,
    pub bytes_transferred: u64,
    pub simulated_ops: usize,
}

enum WorkKind {
    Cpu,
    Io,
    Transfer,
    Inline,
}

struct Completion {
    name: String,
    result: Result<InstructionTelemetry, Box<RuntimeError>>,
}

/// Execute a complete schedule. Caller must not hold the Python GIL.
pub fn execute_schedule(
    schedule: &ExecutableSchedule,
    options: &ExecuteOptions,
    region_cb: Option<RegionCallback>,
    cancel: Option<Arc<AtomicBool>>,
) -> RuntimeResult<ExecuteReport> {
    execute_schedule_ex(schedule, options, region_cb, None, cancel)
}

/// Execute with optional per-instruction Python/native handler (preferred public path).
pub fn execute_schedule_ex(
    schedule: &ExecutableSchedule,
    options: &ExecuteOptions,
    region_cb: Option<RegionCallback>,
    instruction_cb: Option<InstructionCallback>,
    cancel: Option<Arc<AtomicBool>>,
) -> RuntimeResult<ExecuteReport> {
    assert_schedule_valid(schedule)?;
    let t0 = Instant::now();
    if schedule.instructions.is_empty() {
        return Ok(ExecuteReport {
            wall_time_s: t0.elapsed().as_secs_f64(),
            ..ExecuteReport::default()
        });
    }

    // Fast path: dry-run without Python callbacks — no worker-pool spawn.
    if options.dry_run_compute && instruction_cb.is_none() && region_cb.is_none() {
        return execute_dry_run_inline(schedule, t0);
    }

    let by_name: HashMap<String, streamcompiler_core::Instruction> = schedule
        .instructions
        .iter()
        .cloned()
        .map(|i| (i.name.as_str().to_owned(), i))
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

    let allocations = Arc::new(AllocationTable::new());
    let residency = Arc::new(ResidencyStore::new(Arc::clone(&allocations)));
    let completed_events: Arc<Mutex<HashSet<String>>> = Arc::new(Mutex::new(HashSet::new()));
    let alloc_counter = Arc::new(AtomicUsize::new(0));

    let cpu_pool = WorkerPool::new("cpu", options.cpu_workers, options.max_inflight);
    let io_pool = WorkerPool::new("io", options.io_workers, options.max_inflight);
    let xfer_pool = WorkerPool::new("xfer", options.transfer_workers, options.max_inflight);

    let (done_tx, done_rx): (Sender<Completion>, Receiver<Completion>) =
        bounded(options.max_inflight.max(8));

    let mut events = Vec::new();
    let mut inflight = 0usize;
    let mut bytes_read = 0u64;
    let mut bytes_transferred = 0u64;
    let mut simulated_ops = 0usize;
    let mut failure: Option<Box<RuntimeError>> = None;
    let total = schedule.instructions.len();
    let mut finished = 0usize;

    let origin = t0;

    while finished < total {
        if cancel.as_ref().is_some_and(|c| c.load(Ordering::Acquire)) {
            failure = Some(Box::new(RuntimeError::Cancelled));
            break;
        }

        // Drain completions first.
        while let Ok(comp) = done_rx.try_recv() {
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
                    // Free ordered resource slot.
                    let res = tel.resource.clone();
                    resource_busy.remove(&res);
                    if let Some(q) = resource_queues.get_mut(&res) {
                        if let Some(next) = q.pop_front() {
                            ready.push_back(next);
                        }
                    }
                    completed_events.lock().insert(comp.name.clone());
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
            let Some(name) = ready.pop_front() else {
                break;
            };
            let Some(inst) = by_name.get(&name) else {
                continue;
            };

            // Ordered stream gate for device-like resources.
            let res = inst.resource.as_str();
            let ordered = is_ordered_resource(res);
            if ordered && resource_busy.contains(res) {
                resource_queues
                    .entry(res.to_owned())
                    .or_default()
                    .push_back(name);
                continue;
            }
            if ordered {
                resource_busy.insert(res.to_owned());
            }

            let kind = work_kind(inst.opcode);
            let inst = inst.clone();
            let name_c = name.clone();
            let done_tx = done_tx.clone();
            let residency = Arc::clone(&residency);
            let completed_events = Arc::clone(&completed_events);
            let alloc_counter = Arc::clone(&alloc_counter);
            let region_cb = region_cb.clone();
            let instruction_cb = instruction_cb.clone();
            let dry = options.dry_run_compute;

            let job = move || {
                let submitted = origin.elapsed().as_secs_f64();
                let start = origin.elapsed().as_secs_f64();
                let result = if let Some(ref icb) = instruction_cb {
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
                } else {
                    run_instruction(
                        &inst,
                        &residency,
                        &completed_events,
                        &alloc_counter,
                        region_cb.as_ref(),
                        dry,
                    )
                    .map(|simulated| (simulated, inst.nbytes, String::new()))
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

        if inflight == 0 && ready.is_empty() && finished < total {
            // Block for next completion.
            match done_rx.recv() {
                Ok(comp) => {
                    // re-queue for unified handling
                    let _ = done_tx.send(comp);
                }
                Err(_) => {
                    failure = Some(Box::new(RuntimeError::Other("completion channel closed".into())));
                    break;
                }
            }
        } else if inflight > 0 && ready.is_empty() {
            // Wait for at least one completion without busy-wait.
            match done_rx.recv_timeout(std::time::Duration::from_millis(50)) {
                Ok(comp) => {
                    let _ = done_tx.send(comp);
                }
                Err(crossbeam_channel::RecvTimeoutError::Timeout) => {}
                Err(_) => {
                    failure = Some(Box::new(RuntimeError::Other("completion channel closed".into())));
                    break;
                }
            }
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
        peak_activation_bytes: allocations.peak_bytes(),
        allocation_peak_bytes: allocations.peak_bytes(),
        bytes_read,
        bytes_transferred,
        simulated_ops,
    })
}

fn execute_dry_run_inline(
    schedule: &ExecutableSchedule,
    t0: Instant,
) -> RuntimeResult<ExecuteReport> {
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
    let by_name: HashMap<&str, &streamcompiler_core::Instruction> = schedule
        .instructions
        .iter()
        .map(|i| (i.name.as_str(), i))
        .collect();
    let mut events = Vec::with_capacity(schedule.instructions.len());
    while let Some(name) = ready.pop_front() {
        let Some(inst) = by_name.get(name.as_str()) else {
            continue;
        };
        let start = t0.elapsed().as_secs_f64();
        let end = start;
        events.push(InstructionTelemetry {
            name: name.clone(),
            opcode: inst.opcode.to_string(),
            resource: inst.resource.to_string(),
            submitted_s: start,
            start_s: start,
            end_s: end,
            nbytes: inst.nbytes,
            simulated: false,
            notes: "dry_run".into(),
        });
        if let Some(nexts) = dependents.get(&name) {
            for nxt in nexts {
                if let Some(deg) = remaining.get_mut(nxt) {
                    *deg = deg.saturating_sub(1);
                    if *deg == 0 {
                        ready.push_back(nxt.clone());
                    }
                }
            }
        }
    }
    if events.len() != schedule.instructions.len() {
        return Err(Box::new(RuntimeError::Validation(
            "dry-run left unfinished instructions (cycle?)".into(),
        )));
    }
    Ok(ExecuteReport {
        wall_time_s: t0.elapsed().as_secs_f64(),
        events,
        ..ExecuteReport::default()
    })
}

fn enqueue_ready(
    name: &str,
    by_name: &HashMap<String, streamcompiler_core::Instruction>,
    ready: &mut VecDeque<String>,
    resource_queues: &mut HashMap<String, VecDeque<String>>,
    resource_busy: &HashSet<String>,
) {
    let Some(inst) = by_name.get(name) else {
        ready.push_back(name.to_owned());
        return;
    };
    let res = inst.resource.as_str();
    if is_ordered_resource(res) && resource_busy.contains(res) {
        resource_queues
            .entry(res.to_owned())
            .or_default()
            .push_back(name.to_owned());
    } else {
        ready.push_back(name.to_owned());
    }
}

fn is_ordered_resource(res: &str) -> bool {
    let l = res.to_lowercase();
    l.contains("mock")
        || l.contains("cuda")
        || l.contains("rocm")
        || l.contains("gpu")
        || l.contains("accel")
}

fn work_kind(op: Opcode) -> WorkKind {
    match op {
        Opcode::Compute => WorkKind::Cpu,
        Opcode::Prefetch | Opcode::Load | Opcode::Evict => WorkKind::Io,
        Opcode::Transfer => WorkKind::Transfer,
        Opcode::RecordEvent | Opcode::WaitEvent | Opcode::Release => WorkKind::Inline,
    }
}

fn run_instruction(
    inst: &streamcompiler_core::Instruction,
    residency: &ResidencyStore,
    completed_events: &Mutex<HashSet<String>>,
    alloc_counter: &AtomicUsize,
    region_cb: Option<&RegionCallback>,
    dry_run: bool,
) -> RuntimeResult<bool> {
    let mut simulated = false;
    match inst.opcode {
        Opcode::Compute => {
            let inputs: Vec<String> = inst.inputs.iter().map(|t| t.to_string()).collect();
            let outputs: Vec<String> = inst.outputs.iter().map(|t| t.to_string()).collect();
            let region = inst
                .executable_ref
                .as_ref()
                .map(|r| r.as_str())
                .unwrap_or("");
            if dry_run || region_cb.is_none() {
                // Native dry-run / missing callback: account outputs only.
                for out in &inst.outputs {
                    let n = nbytes(inst, out.as_str());
                    let id = next_alloc(alloc_counter);
                    residency
                        .put(
                            TensorId::new(out.as_str()),
                            ResourceId::new(inst.resource.as_str()),
                            id,
                            TensorMetadata {
                                nbytes: n,
                                ..Default::default()
                            },
                            None,
                        )
                        .map_err(|e| inst_err(inst, e.to_string()))?;
                }
                if inst.resource.as_str().contains("mock") {
                    simulated = true;
                    let delay = inst
                        .attributes
                        .get("mock_compute_delay_s")
                        .and_then(|v| match v {
                            streamcompiler_core::AttrValue::Float(f) => Some(*f),
                            streamcompiler_core::AttrValue::Int(i) => Some(*i as f64),
                            _ => None,
                        })
                        .unwrap_or(0.0);
                    if delay > 0.0 {
                        std::thread::sleep(std::time::Duration::from_secs_f64(delay));
                    }
                }
            } else if let Some(cb) = region_cb {
                cb(region, &inputs, &outputs).map_err(|cause| {
                    Box::new(RuntimeError::Instruction {
                        instruction: inst.name.to_string(),
                        opcode: inst.opcode.to_string(),
                        region: Some(region.to_owned()),
                        tensor: None,
                        resource: Some(inst.resource.to_string()),
                        cause,
                    })
                })?;
                for out in &inst.outputs {
                    let n = nbytes(inst, out.as_str());
                    let id = next_alloc(alloc_counter);
                    residency
                        .put(
                            TensorId::new(out.as_str()),
                            ResourceId::new(inst.resource.as_str()),
                            id,
                            TensorMetadata {
                                nbytes: n,
                                ..Default::default()
                            },
                            None,
                        )
                        .map_err(|e| inst_err(inst, e.to_string()))?;
                }
            }
        }
        Opcode::Load | Opcode::Prefetch => {
            let dest = inst
                .destination
                .as_ref()
                .map(|d| d.as_str())
                .unwrap_or(inst.resource.as_str());
            for tid in inst.outputs.iter().chain(inst.inputs.iter()) {
                let n = nbytes(inst, tid.as_str());
                let id = next_alloc(alloc_counter);
                residency
                    .put(
                        TensorId::new(tid.as_str()),
                        ResourceId::new(dest),
                        id,
                        TensorMetadata {
                            nbytes: n,
                            ..Default::default()
                        },
                        None,
                    )
                    .map_err(|e| inst_err(inst, e.to_string()))?;
            }
        }
        Opcode::Transfer => {
            let src = inst.source.as_ref().map(|s| s.as_str()).unwrap_or("");
            let dst = inst
                .destination
                .as_ref()
                .map(|s| s.as_str())
                .unwrap_or(inst.resource.as_str());
            simulated = dst.contains("mock") || src.contains("mock");
            for tid in inst.outputs.iter().chain(inst.inputs.iter()) {
                if let Some(existing) = residency.begin_transfer(
                    &TensorId::new(tid.as_str()),
                    &ResourceId::new(dst),
                    inst.name.as_str(),
                ) {
                    // Share in-progress transfer — wait conceptually by depending on schedule.
                    let _ = existing;
                    continue;
                }
                let n = nbytes(inst, tid.as_str());
                // Ensure source residency exists for accounting.
                if residency
                    .get(&TensorId::new(tid.as_str()), &ResourceId::new(src))
                    .is_err()
                {
                    let id = next_alloc(alloc_counter);
                    residency
                        .put(
                            TensorId::new(tid.as_str()),
                            ResourceId::new(src),
                            id,
                            TensorMetadata {
                                nbytes: n,
                                ..Default::default()
                            },
                            None,
                        )
                        .map_err(|e| inst_err(inst, e.to_string()))?;
                }
                let id = next_alloc(alloc_counter);
                residency
                    .replicate(&TensorId::new(tid.as_str()), ResourceId::new(dst), id, None)
                    .map_err(|e| inst_err(inst, e.to_string()))?;
                residency.end_transfer(&TensorId::new(tid.as_str()), &ResourceId::new(dst));
            }
            if let Some(d) = inst.attributes.get("mock_transfer_delay_s") {
                let delay = match d {
                    streamcompiler_core::AttrValue::Float(f) => *f,
                    streamcompiler_core::AttrValue::Int(i) => *i as f64,
                    _ => 0.0,
                };
                if delay > 0.0 {
                    std::thread::sleep(std::time::Duration::from_secs_f64(delay));
                }
            }
        }
        Opcode::RecordEvent => {
            completed_events.lock().insert(inst.name.to_string());
        }
        Opcode::WaitEvent => {
            let waits = inst
                .attr_str("waits_for")
                .map(str::to_owned)
                .or_else(|| inst.depends_on.first().map(|d| d.as_str().to_owned()));
            if let Some(wf) = waits {
                if !completed_events.lock().contains(&wf) {
                    // Dependency counters guarantee RecordEvent completed; mark present.
                    completed_events.lock().insert(wf);
                }
            }
        }
        Opcode::Evict | Opcode::Release => {
            let res = inst
                .attr_str("release_resource")
                .unwrap_or(inst.resource.as_str());
            for tid in &inst.inputs {
                let _ = residency.release_copy(&TensorId::new(tid.as_str()), &ResourceId::new(res));
            }
        }
    }
    Ok(simulated)
}

fn nbytes(inst: &streamcompiler_core::Instruction, tensor: &str) -> u64 {
    inst.tensor_nbytes()
        .get(tensor)
        .copied()
        .unwrap_or(inst.nbytes)
        .max(1)
}

fn next_alloc(counter: &AtomicUsize) -> AllocationId {
    let n = counter.fetch_add(1, Ordering::Relaxed);
    AllocationId::new(format!("rt-{n}"))
}

fn inst_err(inst: &streamcompiler_core::Instruction, cause: String) -> Box<RuntimeError> {
    Box::new(RuntimeError::Instruction {
        instruction: inst.name.to_string(),
        opcode: inst.opcode.to_string(),
        region: inst.executable_ref.as_ref().map(|r| r.to_string()),
        tensor: inst.inputs.first().map(|t| t.to_string()),
        resource: Some(inst.resource.to_string()),
        cause,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use indexmap::IndexMap;
    use streamcompiler_core::{Instruction, InstructionId, MemoryTier, RegionId};

    #[test]
    fn empty_schedule() {
        let schedule = ExecutableSchedule::new("g", "fp", vec![], vec![]);
        let report = execute_schedule(&schedule, &ExecuteOptions::default(), None, None).unwrap();
        assert_eq!(report.events.len(), 0);
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
                    streamcompiler_core::AttrValue::IntMap(
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
}
