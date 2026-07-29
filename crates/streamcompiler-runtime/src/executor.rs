//! Dependency-counter schedule executor with resource queues and compute callbacks.

use crate::context::NativeExecutionContext;
use crate::error::{RuntimeError, RuntimeResult};
use crate::telemetry::InstructionTelemetry;
use crate::workers::WorkerPool;
use crossbeam_channel::{bounded, Receiver, Sender};
use parking_lot::Mutex;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;
use streamcompiler_core::{assert_schedule_valid, ExecutableSchedule, Opcode};
use streamcompiler_core::{ResourceId, TensorId};
use streamcompiler_memory::TensorMetadata;

/// One ready Compute region for a batched Python invoker call.
#[derive(Clone, Debug)]
pub struct RegionInvocation {
    pub region_id: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
}

/// Callback invoked for one or more ready Compute instructions (single GIL cross).
pub type RegionCallback = Arc<dyn Fn(&[RegionInvocation]) -> Result<(), String> + Send + Sync>;

/// Full instruction dispatch callback: Rust schedules, Python/native backends execute.
/// Returns optional notes string.
pub type InstructionCallback =
    Arc<dyn Fn(&str) -> Result<InstructionCallbackResult, String> + Send + Sync>;

/// Tensor → contiguous host bytes for native activation spill.
pub type DematerializeCallback =
    Arc<dyn Fn(&str) -> Result<(streamcompiler_storage::SpillMeta, Vec<u8>), String> + Send + Sync>;

/// Contiguous host bytes → tensor registration for native activation reload.
pub type MaterializeCallback = Arc<
    dyn Fn(&str, &streamcompiler_storage::SpillMeta, &[u8]) -> Result<(), String> + Send + Sync,
>;

/// Streaming parameter Load: Python acquires pack bytes → torch.Tensor → residency mirror.
/// One call may cover a wave of ready Loads (batched materialization).
/// Each pair is `(tensor_id, destination_resource)` — dest must match the Load instruction.
pub type ParameterLoadCallback =
    Arc<dyn Fn(&[(String, String)]) -> Result<Vec<u64>, String> + Send + Sync>;

/// Drop opaque Python handles when Rust final-releases copies.
/// One call covers a wave of releases (single GIL cross), not one call per tensor.
pub type HandleReleaseCallback =
    Arc<dyn Fn(&[(String, String)]) -> Result<(), String> + Send + Sync>;

/// After Rust Transfer, sync the Python handle table (src → dst).
/// One call covers all tensors in a Transfer instruction (single GIL cross).
pub type CopySyncCallback =
    Arc<dyn Fn(&[(String, String, String, u64)]) -> Result<(), String> + Send + Sync>;

#[derive(Clone, Debug, Default)]
pub struct InstructionCallbackResult {
    pub nbytes: u64,
    pub simulated: bool,
    pub notes: String,
}

#[derive(Clone)]
pub struct ExecuteOptions {
    pub cpu_workers: usize,
    pub io_workers: usize,
    pub transfer_workers: usize,
    pub max_inflight: usize,
    /// When true, Compute is a timed no-op (native microbench / empty schedule).
    pub dry_run_compute: bool,
    pub dematerialize: Option<DematerializeCallback>,
    pub materialize: Option<MaterializeCallback>,
    pub parameter_load: Option<ParameterLoadCallback>,
    pub handle_release: Option<HandleReleaseCallback>,
    pub copy_sync: Option<CopySyncCallback>,
}

impl std::fmt::Debug for ExecuteOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExecuteOptions")
            .field("cpu_workers", &self.cpu_workers)
            .field("io_workers", &self.io_workers)
            .field("transfer_workers", &self.transfer_workers)
            .field("max_inflight", &self.max_inflight)
            .field("dry_run_compute", &self.dry_run_compute)
            .field("dematerialize", &self.dematerialize.is_some())
            .field("materialize", &self.materialize.is_some())
            .field("parameter_load", &self.parameter_load.is_some())
            .field("handle_release", &self.handle_release.is_some())
            .field("copy_sync", &self.copy_sync.is_some())
            .finish()
    }
}

impl Default for ExecuteOptions {
    fn default() -> Self {
        Self {
            cpu_workers: 4,
            io_workers: 2,
            transfer_workers: 2,
            max_inflight: 64,
            dry_run_compute: false,
            dematerialize: None,
            materialize: None,
            parameter_load: None,
            handle_release: None,
            copy_sync: None,
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
    /// Stream/engine order key used for serialization (not device name).
    order_key: Option<String>,
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
    let cancel = cancel.unwrap_or_else(|| Arc::new(AtomicBool::new(false)));
    let ctx = NativeExecutionContext::with_cancel(Arc::clone(&cancel));
    execute_schedule_with_context(schedule, options, region_cb, instruction_cb, ctx)
}

/// Execute against a caller-owned [`NativeExecutionContext`].
///
/// Python creates one context per forward, mirrors residency into it, then
/// passes the same context here so Load/Release/Transfer and Compute share
/// one residency store, event table, and allocation table.
pub fn execute_schedule_with_context(
    schedule: &ExecutableSchedule,
    options: &ExecuteOptions,
    region_cb: Option<RegionCallback>,
    instruction_cb: Option<InstructionCallback>,
    ctx: Arc<NativeExecutionContext>,
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
        return execute_dry_run_inline(schedule, &ctx, t0);
    }

    // Hybrid: Compute via region_cb; Load/Prefetch/Release/Evict native or via instruction_cb.
    // Prefetch queues onto a background I/O worker — inline Prefetch-before-Compute
    // overlaps without spawning per-forward thread pools. Pool only when Transfer
    // can run beside Compute (cross-device).
    let width = max_ready_width(schedule);
    let needs_transfer_pool = width > 1
        && schedule
            .instructions
            .iter()
            .any(|i| i.opcode == Opcode::Transfer);
    if let (Some(ref rcb), Some(ref icb)) = (&region_cb, &instruction_cb) {
        if !needs_transfer_pool {
            return execute_region_cb_inline(schedule, options, rcb, Some(icb), &ctx, t0);
        }
    } else if let Some(ref icb) = instruction_cb {
        if !needs_transfer_pool {
            return execute_instruction_cb_inline(schedule, icb, &ctx, t0);
        }
    } else if let Some(ref rcb) = region_cb {
        if !needs_transfer_pool {
            return execute_region_cb_inline(schedule, options, rcb, None, &ctx, t0);
        }
    }

    execute_schedule_pooled(schedule, options, region_cb, instruction_cb, ctx, t0)
}

fn max_ready_width(schedule: &ExecutableSchedule) -> usize {
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
    let mut peak = ready.len();
    while let Some(name) = ready.pop_front() {
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
        peak = peak.max(ready.len());
    }
    peak
}

fn needs_python_io(inst: &streamcompiler_core::Instruction, opts: &ExecuteOptions) -> bool {
    // Only ops that still need a Python instruction body when an instruction
    // callback is installed. Prefetch/Release/Evict/Transfer are native.
    // Activation spill/reload use dematerialize/materialize.
    // Streaming parameter Load uses parameter_load (materialization, not I/O callback).
    let kind = inst.attr_str("kind").unwrap_or("");
    match inst.opcode {
        Opcode::Load => {
            if kind == "activation_reload" && opts.materialize.is_some() {
                return false;
            }
            if kind == "parameter_materialize" && opts.parameter_load.is_some() {
                return false;
            }
            // Legacy: only when neither native path is available.
            kind == "activation_reload" || kind == "parameter_materialize"
        }
        Opcode::Evict => {
            if kind == "activation_spill" && opts.dematerialize.is_some() {
                return false;
            }
            // parameter_evict is native residency release.
            kind == "activation_spill"
        }
        Opcode::Prefetch | Opcode::Release => false,
        _ => false,
    }
}

fn execute_region_cb_inline(
    schedule: &ExecutableSchedule,
    options: &ExecuteOptions,
    region_cb: &RegionCallback,
    instruction_cb: Option<&InstructionCallback>,
    ctx: &NativeExecutionContext,
    t0: Instant,
) -> RuntimeResult<ExecuteReport> {
    let residency = ctx.residency();
    let cancel = ctx.cancel_flag();

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
    let index_of: HashMap<&str, usize> = schedule
        .instructions
        .iter()
        .enumerate()
        .map(|(i, inst)| (inst.name.as_str(), i))
        .collect();

    let mut events = Vec::with_capacity(schedule.instructions.len());
    let mut bytes_read = 0u64;
    let mut bytes_transferred = 0u64;
    let mut simulated_ops = 0usize;
    let origin = t0;

    while !ready.is_empty() {
        if cancel.load(Ordering::Acquire) {
            return Err(Box::new(RuntimeError::Cancelled));
        }
        let mut compute_names: Vec<String> = Vec::new();
        let mut other: VecDeque<String> = VecDeque::new();
        while let Some(name) = ready.pop_front() {
            let Some(inst) = by_name.get(name.as_str()) else {
                continue;
            };
            if inst.opcode == Opcode::Compute {
                compute_names.push(name);
            } else {
                other.push_back(name);
            }
        }
        compute_names.sort_by_key(|n| index_of.get(n.as_str()).copied().unwrap_or(usize::MAX));

        // Wave-batch ready Releases / parameter_evicts before other I/O (one GIL).
        if options.handle_release.is_some() {
            let mut release_insts: Vec<&streamcompiler_core::Instruction> = Vec::new();
            let mut release_names: Vec<String> = Vec::new();
            let mut rest = VecDeque::new();
            while let Some(name) = other.pop_front() {
                let Some(inst) = by_name.get(name.as_str()) else {
                    continue;
                };
                if is_batched_handle_release(inst) {
                    release_names.push(name);
                    release_insts.push(*inst);
                } else {
                    rest.push_back(name);
                }
            }
            other = rest;
            if !release_insts.is_empty() {
                let teles = run_release_wave(&release_insts, ctx, options, origin)?;
                for (tel, name) in teles.into_iter().zip(release_names.iter()) {
                    if matches!(tel.opcode.as_str(), "Load" | "Prefetch") {
                        bytes_read += tel.nbytes;
                    }
                    if tel.simulated {
                        simulated_ops += 1;
                    }
                    events.push(tel);
                    if let Some(nexts) = dependents.get(name) {
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
            }
        }

        // Wave-batch ready parameter Loads (one GIL for the whole wave).
        if options.parameter_load.is_some() {
            let mut load_insts: Vec<&streamcompiler_core::Instruction> = Vec::new();
            let mut load_names: Vec<String> = Vec::new();
            let mut rest = VecDeque::new();
            while let Some(name) = other.pop_front() {
                let Some(inst) = by_name.get(name.as_str()) else {
                    continue;
                };
                if is_batched_parameter_load(inst) {
                    load_names.push(name);
                    load_insts.push(*inst);
                } else {
                    rest.push_back(name);
                }
            }
            other = rest;
            if !load_insts.is_empty() {
                let teles = run_parameter_load_wave(&load_insts, ctx, options, origin)?;
                for (tel, name) in teles.into_iter().zip(load_names.iter()) {
                    bytes_read += tel.nbytes;
                    events.push(tel);
                    if let Some(nexts) = dependents.get(name) {
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
            }
        }

        // Prefetch/Transfer/Load first so background I/O overlaps the Compute wave.
        while let Some(name) = other.pop_front() {
            if cancel.load(Ordering::Acquire) {
                return Err(Box::new(RuntimeError::Cancelled));
            }
            let Some(inst) = by_name.get(name.as_str()) else {
                continue;
            };
            let submitted = origin.elapsed().as_secs_f64();
            let start = origin.elapsed().as_secs_f64();
            let (nbytes_out, simulated, notes) = if needs_python_io(inst, options) {
                if let Some(icb) = instruction_cb {
                    let outcome = icb(inst.name.as_str()).map_err(|cause| {
                        Box::new(RuntimeError::Instruction {
                            instruction: inst.name.to_string(),
                            opcode: inst.opcode.to_string(),
                            region: inst.executable_ref.as_ref().map(|r| r.to_string()),
                            tensor: inst.inputs.first().map(|t| t.to_string()),
                            resource: Some(inst.resource.to_string()),
                            cause,
                        })
                    })?;
                    (outcome.nbytes, outcome.simulated, outcome.notes)
                } else {
                    let notes = load_notes(inst, &residency);
                    let simulated = run_instruction(inst, ctx, Some(region_cb), false, options)?;
                    (inst.nbytes, simulated, notes)
                }
            } else {
                let notes = load_notes(inst, &residency);
                let simulated = run_instruction(inst, ctx, Some(region_cb), false, options)?;
                (inst.nbytes, simulated, notes)
            };
            let end = origin.elapsed().as_secs_f64();
            if matches!(inst.opcode, Opcode::Load | Opcode::Prefetch) {
                bytes_read += nbytes_out;
            }
            if matches!(inst.opcode, Opcode::Transfer) {
                bytes_transferred += nbytes_out;
            }
            if simulated {
                simulated_ops += 1;
            }
            events.push(InstructionTelemetry {
                name: name.clone(),
                opcode: inst.opcode.to_string(),
                resource: inst.resource.to_string(),
                submitted_s: submitted,
                start_s: start,
                end_s: end,
                nbytes: nbytes_out,
                simulated,
                notes,
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

        if !compute_names.is_empty() {
            let mut invocations: Vec<RegionInvocation> = Vec::with_capacity(compute_names.len());
            let mut batch_meta: Vec<(&streamcompiler_core::Instruction, f64, f64)> =
                Vec::with_capacity(compute_names.len());
            for name in &compute_names {
                let Some(inst) = by_name.get(name.as_str()) else {
                    continue;
                };
                let submitted = origin.elapsed().as_secs_f64();
                let region = inst
                    .executable_ref
                    .as_ref()
                    .map(|r| r.as_str())
                    .unwrap_or("");
                invocations.push(RegionInvocation {
                    region_id: region.to_owned(),
                    inputs: inst.inputs.iter().map(|t| t.to_string()).collect(),
                    outputs: inst.outputs.iter().map(|t| t.to_string()).collect(),
                });
                batch_meta.push((*inst, submitted, submitted));
            }
            region_cb(&invocations).map_err(|cause| {
                let region = invocations
                    .first()
                    .map(|i| i.region_id.clone())
                    .unwrap_or_default();
                Box::new(RuntimeError::Instruction {
                    instruction: compute_names
                        .first()
                        .cloned()
                        .unwrap_or_else(|| "compute_batch".into()),
                    opcode: "Compute".into(),
                    region: Some(region),
                    tensor: None,
                    resource: None,
                    cause,
                })
            })?;
            let batch_note = if compute_names.len() > 1 {
                format!("region_callback_batch:{}", compute_names.len())
            } else {
                "region_callback".into()
            };
            for (inst, submitted, start) in batch_meta {
                let mut simulated = false;
                // Simulated accelerator after region body (same as run_instruction_body).
                if inst.resource.as_str().contains("mock") {
                    simulate_mock_compute(inst, ctx)?;
                    simulated = true;
                }
                let end = origin.elapsed().as_secs_f64();
                for out in &inst.outputs {
                    let tensor = TensorId::new(out.as_str());
                    let resource = ResourceId::new(inst.resource.as_str());
                    if residency.get(&tensor, &resource).is_ok() {
                        continue;
                    }
                    let n = nbytes(inst, out.as_str());
                    let id = ctx.next_alloc_id();
                    residency
                        .put(
                            tensor,
                            resource,
                            id,
                            TensorMetadata {
                                nbytes: n,
                                ..Default::default()
                            },
                            None,
                        )
                        .map_err(|e| inst_err(inst, e.to_string()))?;
                }
                if simulated {
                    simulated_ops += 1;
                }
                events.push(InstructionTelemetry {
                    name: inst.name.as_str().to_owned(),
                    opcode: inst.opcode.to_string(),
                    resource: inst.resource.to_string(),
                    submitted_s: submitted,
                    start_s: start,
                    end_s: end,
                    nbytes: inst.nbytes,
                    simulated,
                    notes: batch_note.clone(),
                });
                if let Some(nexts) = dependents.get(inst.name.as_str()) {
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
        }
    }

    if events.len() != schedule.instructions.len() {
        return Err(Box::new(RuntimeError::Other(format!(
            "region-cb schedule left unfinished: done={} total={}",
            events.len(),
            schedule.instructions.len()
        ))));
    }

    let _ = residency;
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

fn execute_instruction_cb_inline(
    schedule: &ExecutableSchedule,
    icb: &InstructionCallback,
    ctx: &NativeExecutionContext,
    t0: Instant,
) -> RuntimeResult<ExecuteReport> {
    let cancel = ctx.cancel_flag();
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
    let mut bytes_read = 0u64;
    let mut bytes_transferred = 0u64;
    let mut simulated_ops = 0usize;
    let origin = t0;

    while let Some(name) = ready.pop_front() {
        if cancel.load(Ordering::Acquire) {
            return Err(Box::new(RuntimeError::Cancelled));
        }
        let Some(inst) = by_name.get(name.as_str()) else {
            continue;
        };
        let submitted = origin.elapsed().as_secs_f64();
        let start = origin.elapsed().as_secs_f64();
        let outcome = icb(inst.name.as_str()).map_err(|cause| {
            Box::new(RuntimeError::Instruction {
                instruction: inst.name.to_string(),
                opcode: inst.opcode.to_string(),
                region: inst.executable_ref.as_ref().map(|r| r.to_string()),
                tensor: inst.inputs.first().map(|t| t.to_string()),
                resource: Some(inst.resource.to_string()),
                cause,
            })
        })?;
        let end = origin.elapsed().as_secs_f64();
        if matches!(inst.opcode, Opcode::Load | Opcode::Prefetch) {
            bytes_read += outcome.nbytes;
        }
        if matches!(inst.opcode, Opcode::Transfer) {
            bytes_transferred += outcome.nbytes;
        }
        if outcome.simulated {
            simulated_ops += 1;
        }
        events.push(InstructionTelemetry {
            name: name.clone(),
            opcode: inst.opcode.to_string(),
            resource: inst.resource.to_string(),
            submitted_s: submitted,
            start_s: start,
            end_s: end,
            nbytes: outcome.nbytes,
            simulated: outcome.simulated,
            notes: outcome.notes,
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
        return Err(Box::new(RuntimeError::Other(format!(
            "inline schedule left unfinished: done={} total={}",
            events.len(),
            schedule.instructions.len()
        ))));
    }

    Ok(ExecuteReport {
        wall_time_s: t0.elapsed().as_secs_f64(),
        events,
        bytes_read,
        bytes_transferred,
        simulated_ops,
        allocation_peak_bytes: ctx.peak_bytes(),
        peak_activation_bytes: ctx.peak_bytes(),
    })
}

fn execute_schedule_pooled(
    schedule: &ExecutableSchedule,
    options: &ExecuteOptions,
    region_cb: Option<RegionCallback>,
    instruction_cb: Option<InstructionCallback>,
    ctx: Arc<NativeExecutionContext>,
    t0: Instant,
) -> RuntimeResult<ExecuteReport> {
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
    let total = schedule.instructions.len();
    let mut finished = 0usize;

    let origin = t0;

    while finished < total {
        if cancel.load(Ordering::Acquire) {
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
                    let wave_insts: Vec<&streamcompiler_core::Instruction> = release_names
                        .iter()
                        .filter_map(|n| by_name.get(n))
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
                    let wave_insts: Vec<&streamcompiler_core::Instruction> =
                        load_names.iter().filter_map(|n| by_name.get(n)).collect();
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
                    .get(*n)
                    .map(|i| {
                        matches!(
                            i.opcode,
                            Opcode::Prefetch | Opcode::Transfer | Opcode::Load | Opcode::Evict
                        ) && !is_batched_handle_release(i)
                    })
                    .unwrap_or(false)
            }) {
                let name = name.clone();
                ready.retain(|n| n != &name);
                let Some(inst) = by_name.get(&name) else {
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
                        .filter_map(|n| by_name.get(n).cloned())
                        .collect();
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
            let Some(inst) = by_name.get(&name) else {
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

        if inflight == 0 && ready.is_empty() && finished < total {
            // Block for next completion.
            match done_rx.recv() {
                Ok(comp) => {
                    // re-queue for unified handling
                    let _ = done_tx.send(comp);
                }
                Err(_) => {
                    failure = Some(Box::new(RuntimeError::Other(
                        "completion channel closed".into(),
                    )));
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
                    failure = Some(Box::new(RuntimeError::Other(
                        "completion channel closed".into(),
                    )));
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
        peak_activation_bytes: ctx.peak_bytes(),
        allocation_peak_bytes: ctx.peak_bytes(),
        bytes_read,
        bytes_transferred,
        simulated_ops,
    })
}

fn execute_dry_run_inline(
    schedule: &ExecutableSchedule,
    ctx: &NativeExecutionContext,
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
    let _ = ctx;
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
    if let Some(key) = order_key(inst) {
        if resource_busy.contains(&key) {
            resource_queues
                .entry(key)
                .or_default()
                .push_back(name.to_owned());
            return;
        }
    }
    ready.push_back(name.to_owned());
}

/// Pull ready Release / parameter_evict ops; leave everything else in `ready`.
fn take_ready_release_wave(
    ready: &mut VecDeque<String>,
    by_name: &HashMap<String, streamcompiler_core::Instruction>,
) -> Vec<String> {
    let mut wave = Vec::new();
    let mut rest = VecDeque::new();
    while let Some(name) = ready.pop_front() {
        let Some(inst) = by_name.get(&name) else {
            continue;
        };
        if is_batched_handle_release(inst) {
            wave.push(name);
        } else {
            rest.push_back(name);
        }
    }
    *ready = rest;
    wave
}

/// Pull ready parameter_materialize Loads; leave everything else in `ready`.
fn take_ready_parameter_load_wave(
    ready: &mut VecDeque<String>,
    by_name: &HashMap<String, streamcompiler_core::Instruction>,
) -> Vec<String> {
    let mut wave = Vec::new();
    let mut rest = VecDeque::new();
    while let Some(name) = ready.pop_front() {
        let Some(inst) = by_name.get(&name) else {
            continue;
        };
        if is_batched_parameter_load(inst) {
            wave.push(name);
        } else {
            rest.push_back(name);
        }
    }
    *ready = rest;
    wave
}

fn is_batched_parameter_load(inst: &streamcompiler_core::Instruction) -> bool {
    inst.opcode == Opcode::Load && inst.attr_str("kind").unwrap_or("") == "parameter_materialize"
}

/// Run a parameter-load wave: residency checks per-op, one batched callback.
fn run_parameter_load_wave(
    wave_insts: &[&streamcompiler_core::Instruction],
    ctx: &NativeExecutionContext,
    options: &ExecuteOptions,
    origin: Instant,
) -> RuntimeResult<Vec<InstructionTelemetry>> {
    let residency = ctx.residency();
    let Some(pload) = options.parameter_load.as_ref() else {
        // No callback — fall back to per-instruction path.
        let mut teles = Vec::with_capacity(wave_insts.len());
        for inst in wave_insts {
            let submitted = origin.elapsed().as_secs_f64();
            let start = origin.elapsed().as_secs_f64();
            let simulated = run_instruction(inst, ctx, None, false, options)?;
            let end = origin.elapsed().as_secs_f64();
            teles.push(InstructionTelemetry {
                name: inst.name.as_str().to_owned(),
                opcode: inst.opcode.to_string(),
                resource: inst.resource.to_string(),
                submitted_s: submitted,
                start_s: start,
                end_s: end,
                nbytes: inst.nbytes,
                simulated,
                notes: String::from("native_data_plane"),
            });
        }
        return Ok(teles);
    };

    // Collect missing (tensor, dest) across the wave; one GIL for all.
    let mut need: Vec<(String, String)> = Vec::new();
    for inst in wave_insts {
        let dest = inst
            .destination
            .as_ref()
            .map(|d| d.as_str())
            .unwrap_or(inst.resource.as_str());
        for tid in inst.outputs.iter().chain(inst.inputs.iter()) {
            let tensor = TensorId::new(tid.as_str());
            let resource = ResourceId::new(dest);
            if residency.get(&tensor, &resource).is_ok() {
                continue;
            }
            if !need
                .iter()
                .any(|(t, d)| t == tid.as_str() && d.as_str() == dest)
            {
                need.push((tid.as_str().to_owned(), dest.to_owned()));
            }
        }
    }
    if !need.is_empty() {
        let sizes = pload(&need).map_err(|e| {
            Box::new(RuntimeError::Other(format!("batched parameter_load: {e}")))
                as Box<RuntimeError>
        })?;
        if sizes.len() != need.len() {
            return Err(Box::new(RuntimeError::Other(format!(
                "parameter_load returned {} sizes for {} tensors",
                sizes.len(),
                need.len()
            ))));
        }
        // Python must mirror onto the exact Load destination — refuse invent.
        for ((tid, dest), _n) in need.iter().zip(sizes) {
            let tensor = TensorId::new(tid.as_str());
            let resource = ResourceId::new(dest.as_str());
            let copy = residency.get(&tensor, &resource).map_err(|_| {
                Box::new(RuntimeError::Other(format!(
                    "parameter_load did not materialize {tid} on {dest} (refuse invent)"
                ))) as Box<RuntimeError>
            })?;
            if copy.external_handle.is_none() {
                return Err(Box::new(RuntimeError::Other(format!(
                    "parameter_load left {tid} on {dest} without opaque handle (refuse invent)"
                ))));
            }
        }
    }

    let mut teles = Vec::with_capacity(wave_insts.len());
    for inst in wave_insts {
        let submitted = origin.elapsed().as_secs_f64();
        let t = origin.elapsed().as_secs_f64();
        teles.push(InstructionTelemetry {
            name: inst.name.as_str().to_owned(),
            opcode: inst.opcode.to_string(),
            resource: inst.resource.to_string(),
            submitted_s: submitted,
            start_s: t,
            end_s: t,
            nbytes: inst.nbytes,
            simulated: false,
            notes: String::from("native_data_plane_batched_pload"),
        });
    }
    Ok(teles)
}

/// Run a release wave with residency work per-op, then one batched handle_release callback.
fn run_release_wave(
    wave_insts: &[&streamcompiler_core::Instruction],
    ctx: &NativeExecutionContext,
    options: &ExecuteOptions,
    origin: Instant,
) -> RuntimeResult<Vec<InstructionTelemetry>> {
    let collected: Arc<Mutex<Vec<(String, String)>>> = Arc::new(Mutex::new(Vec::new()));
    let mut wave_opts = options.clone();
    if options.handle_release.is_some() {
        let bucket = Arc::clone(&collected);
        wave_opts.handle_release = Some(Arc::new(move |pairs: &[(String, String)]| {
            bucket.lock().extend_from_slice(pairs);
            Ok(())
        }) as HandleReleaseCallback);
    }
    let mut teles = Vec::with_capacity(wave_insts.len());
    for inst in wave_insts {
        let submitted = origin.elapsed().as_secs_f64();
        let start = origin.elapsed().as_secs_f64();
        let simulated = run_instruction(inst, ctx, None, false, &wave_opts)?;
        let end = origin.elapsed().as_secs_f64();
        teles.push(InstructionTelemetry {
            name: inst.name.as_str().to_owned(),
            opcode: inst.opcode.to_string(),
            resource: inst.resource.to_string(),
            submitted_s: submitted,
            start_s: start,
            end_s: end,
            nbytes: inst.nbytes,
            simulated,
            notes: String::from("native_data_plane"),
        });
    }
    let pairs = collected.lock().clone();
    if let Some(href) = options.handle_release.as_ref() {
        if !pairs.is_empty() {
            href(&pairs).map_err(|e| {
                Box::new(RuntimeError::Other(format!("batched handle_release: {e}")))
                    as Box<RuntimeError>
            })?;
        }
    }
    Ok(teles)
}

/// Drain ready Computes that are not stream/engine-blocked into one wave.
/// Marks their order keys busy. Returns `None` when no Compute is launchable.
fn take_ready_compute_wave(
    ready: &mut VecDeque<String>,
    by_name: &HashMap<String, streamcompiler_core::Instruction>,
    resource_busy: &mut HashSet<String>,
    resource_queues: &mut HashMap<String, VecDeque<String>>,
) -> Option<Vec<String>> {
    let mut wave = Vec::new();
    let mut deferred = VecDeque::new();
    while let Some(name) = ready.pop_front() {
        let Some(inst) = by_name.get(&name) else {
            continue;
        };
        if inst.opcode != Opcode::Compute {
            deferred.push_back(name);
            continue;
        }
        if let Some(key) = order_key(inst) {
            if resource_busy.contains(&key) {
                resource_queues.entry(key).or_default().push_back(name);
                continue;
            }
            resource_busy.insert(key);
        }
        wave.push(name);
    }
    while let Some(n) = deferred.pop_back() {
        ready.push_front(n);
    }
    if wave.is_empty() {
        None
    } else {
        Some(wave)
    }
}

/// Serialize on stream_id when present; never whole-device when streams differ.
fn order_key(inst: &streamcompiler_core::Instruction) -> Option<String> {
    if !is_ordered_resource(inst.resource.as_str()) {
        return None;
    }
    if let Some(ref sid) = inst.stream_id {
        if !sid.as_str().is_empty() {
            return Some(format!("stream:{}", sid.as_str()));
        }
    }
    if let Some(ref eng) = inst.copy_engine_id {
        if !eng.as_str().is_empty() {
            return Some(format!("engine:{}", eng.as_str()));
        }
    }
    Some(format!("resource:{}", inst.resource.as_str()))
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

fn acquire_capacity(
    ctx: &NativeExecutionContext,
    kind: &str,
    id: &str,
    max_concurrent: u32,
) -> RuntimeResult<()> {
    loop {
        if ctx.is_cancelled() {
            return Err(Box::new(RuntimeError::Cancelled));
        }
        let ok = ctx.with_resources(|rs| {
            let cap = match kind {
                "copy" => rs.ensure_copy_engine(id, max_concurrent),
                "io" => rs.ensure_io_queue(id, max_concurrent),
                _ => return true,
            };
            cap.try_acquire()
        });
        if ok {
            return Ok(());
        }
        std::thread::yield_now();
    }
}

fn acquire_link(ctx: &NativeExecutionContext, link_id: &str, nbytes: u64) -> RuntimeResult<()> {
    loop {
        if ctx.is_cancelled() {
            return Err(Box::new(RuntimeError::Cancelled));
        }
        let ok = ctx.with_resources(|rs| {
            let link = rs.links.entry(link_id.to_owned()).or_default();
            // One transfer per link at a time (contention).
            if link.bytes_in_flight > 0 {
                return false;
            }
            link.bytes_in_flight = nbytes.max(1);
            true
        });
        if ok {
            return Ok(());
        }
        std::thread::yield_now();
    }
}

fn run_instruction(
    inst: &streamcompiler_core::Instruction,
    ctx: &NativeExecutionContext,
    region_cb: Option<&RegionCallback>,
    dry_run: bool,
    options: &ExecuteOptions,
) -> RuntimeResult<bool> {
    let residency = ctx.residency();
    let mut simulated = false;
    if let Some(ref sid) = inst.stream_id {
        ctx.with_resources(|rs| rs.note_stream_submit(sid.as_str(), 0.0));
    }
    if let Some(ref eng) = inst.copy_engine_id {
        acquire_capacity(ctx, "copy", eng.as_str(), 2)?;
    }
    if let Some(ref qid) = inst.io_queue_id {
        acquire_capacity(ctx, "io", qid.as_str(), 2)?;
    }
    if let Some(ref link) = inst.link_id {
        acquire_link(ctx, link.as_str(), inst.nbytes)?;
    }
    let result = run_instruction_body(
        inst,
        ctx,
        &residency,
        region_cb,
        dry_run,
        options,
        &mut simulated,
    );
    if let Some(ref link) = inst.link_id {
        ctx.with_resources(|rs| {
            if let Some(st) = rs.links.get_mut(link.as_str()) {
                st.bytes_in_flight = 0;
            }
        });
    }
    if let Some(ref qid) = inst.io_queue_id {
        ctx.with_resources(|rs| {
            if let Some(q) = rs.io_queues.get_mut(qid.as_str()) {
                q.release();
            }
        });
    }
    if let Some(ref eng) = inst.copy_engine_id {
        ctx.with_resources(|rs| {
            if let Some(cap) = rs.copy_engines.get_mut(eng.as_str()) {
                cap.release();
            }
        });
    }
    if let Some(ref sid) = inst.stream_id {
        ctx.with_resources(|rs| rs.note_stream_complete(sid.as_str()));
    }
    result?;
    Ok(simulated)
}

fn attr_delay_s(inst: &streamcompiler_core::Instruction, key: &str) -> Option<f64> {
    inst.attributes.get(key).and_then(|v| match v {
        streamcompiler_core::AttrValue::Float(f) => Some(*f),
        streamcompiler_core::AttrValue::Int(i) => Some(*i as f64),
        _ => None,
    })
}

/// Public mock path: Rust virtual backend (buffers/streams/pending events), not Python MockStream.
fn simulate_mock_compute(
    inst: &streamcompiler_core::Instruction,
    ctx: &NativeExecutionContext,
) -> RuntimeResult<()> {
    let delay = attr_delay_s(inst, "mock_compute_delay_s").unwrap_or(0.0);
    if delay <= 0.0 && !inst.resource.as_str().contains("mock") {
        return Ok(());
    }
    let stream = inst
        .stream_id
        .as_ref()
        .map(|s| s.as_str().to_owned())
        .unwrap_or_else(|| format!("{}::compute0", inst.resource.as_str()));
    let be = ctx.virtual_backend(inst.resource.as_str());
    be.run_compute(&stream, delay)
        .map_err(|e| inst_err(inst, format!("virtual backend compute (simulated): {e}")))
}

fn simulate_mock_transfer(
    inst: &streamcompiler_core::Instruction,
    ctx: &NativeExecutionContext,
    dst: &str,
    nbytes: u64,
) -> RuntimeResult<()> {
    let delay = attr_delay_s(inst, "mock_transfer_delay_s");
    if delay.is_none() && !dst.contains("mock") && !inst.resource.as_str().contains("mock") {
        return Ok(());
    }
    let stream = inst
        .stream_id
        .as_ref()
        .map(|s| s.as_str().to_owned())
        .unwrap_or_else(|| format!("{dst}::copy0"));
    let resource = if dst.contains("mock") {
        dst
    } else {
        inst.resource.as_str()
    };
    let be = ctx.virtual_backend(resource);
    be.run_transfer(&stream, nbytes.max(1) as usize, delay)
        .map_err(|e| inst_err(inst, format!("virtual backend transfer (simulated): {e}")))
}

fn run_instruction_body(
    inst: &streamcompiler_core::Instruction,
    ctx: &NativeExecutionContext,
    residency: &streamcompiler_memory::ResidencyStore,
    region_cb: Option<&RegionCallback>,
    dry_run: bool,
    options: &ExecuteOptions,
    simulated: &mut bool,
) -> RuntimeResult<()> {
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
                    let id = ctx.next_alloc_id();
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
                    *simulated = true;
                    simulate_mock_compute(inst, ctx)?;
                }
            } else if let Some(cb) = region_cb {
                let inv = RegionInvocation {
                    region_id: region.to_owned(),
                    inputs,
                    outputs,
                };
                cb(std::slice::from_ref(&inv)).map_err(|cause| {
                    Box::new(RuntimeError::Instruction {
                        instruction: inst.name.to_string(),
                        opcode: inst.opcode.to_string(),
                        region: Some(region.to_owned()),
                        tensor: None,
                        resource: Some(inst.resource.to_string()),
                        cause,
                    })
                })?;
                // Simulated accelerator after real region body (overlap tests).
                if inst.resource.as_str().contains("mock") {
                    *simulated = true;
                    simulate_mock_compute(inst, ctx)?;
                }
                for out in &inst.outputs {
                    let tensor = TensorId::new(out.as_str());
                    let resource = ResourceId::new(inst.resource.as_str());
                    // Python region callback already mirrored outputs into the
                    // shared residency store — keep those opaque handles.
                    if residency.get(&tensor, &resource).is_ok() {
                        continue;
                    }
                    let n = nbytes(inst, out.as_str());
                    let id = ctx.next_alloc_id();
                    residency
                        .put(
                            tensor,
                            resource,
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
            let kind = inst.attr_str("kind").unwrap_or("");
            if kind == "activation_reload" {
                return native_activation_reload(inst, ctx, residency, options);
            }
            if inst.opcode == Opcode::Prefetch {
                if let Some(store) = ctx.streaming_store() {
                    let keys: Vec<String> = inst
                        .outputs
                        .iter()
                        .chain(inst.inputs.iter())
                        .map(|t| ctx.pack_key(t.as_str()))
                        .collect::<std::collections::HashSet<_>>()
                        .into_iter()
                        .collect();
                    store.prefetch(&keys);
                    return Ok(());
                }
                // Resident schedules: Prefetch is a no-op (weights already mapped).
                return Ok(());
            }
            // Load
            if kind == "parameter_materialize" {
                if let Some(pload) = options.parameter_load.as_ref() {
                    let dest = inst
                        .destination
                        .as_ref()
                        .map(|d| d.as_str())
                        .unwrap_or(inst.resource.as_str());
                    let mut need: Vec<(String, String)> = Vec::new();
                    for tid in inst.outputs.iter().chain(inst.inputs.iter()) {
                        let tensor = TensorId::new(tid.as_str());
                        let resource = ResourceId::new(dest);
                        if residency.get(&tensor, &resource).is_ok() {
                            continue;
                        }
                        if !need.iter().any(|(t, _)| t == tid.as_str()) {
                            need.push((tid.as_str().to_owned(), dest.to_owned()));
                        }
                    }
                    if !need.is_empty() {
                        let sizes = pload(&need).map_err(|e| inst_err(inst, e))?;
                        if sizes.len() != need.len() {
                            return Err(inst_err(
                                inst,
                                format!(
                                    "parameter_load returned {} sizes for {} tensors",
                                    sizes.len(),
                                    need.len()
                                ),
                            ));
                        }
                        for ((tid, d), _n) in need.iter().zip(sizes) {
                            let tensor = TensorId::new(tid.as_str());
                            let resource = ResourceId::new(d.as_str());
                            let copy = residency.get(&tensor, &resource).map_err(|_| {
                                inst_err(
                                    inst,
                                    format!(
                                        "parameter_load did not materialize {tid} on {d} (refuse invent)"
                                    ),
                                )
                            })?;
                            if copy.external_handle.is_none() {
                                return Err(inst_err(
                                    inst,
                                    format!(
                                        "parameter_load left {tid} on {d} without opaque handle (refuse invent)"
                                    ),
                                ));
                            }
                        }
                    }
                    return Ok(());
                }
            }
            let dest = inst
                .destination
                .as_ref()
                .map(|d| d.as_str())
                .unwrap_or(inst.resource.as_str());
            for tid in inst.outputs.iter().chain(inst.inputs.iter()) {
                let tensor = TensorId::new(tid.as_str());
                let resource = ResourceId::new(dest);
                // Persistent initial residency: Python registered the copy before
                // schedule execution. Verify at the real Load position — do not
                // invent a fake prematerialized Load event.
                if residency.get(&tensor, &resource).is_ok() {
                    continue;
                }
                if dry_run {
                    let n = nbytes(inst, tid.as_str());
                    let id = ctx.next_alloc_id();
                    residency
                        .put(
                            tensor,
                            resource,
                            id,
                            TensorMetadata {
                                nbytes: n,
                                ..Default::default()
                            },
                            None,
                        )
                        .map_err(|e| inst_err(inst, e.to_string()))?;
                    continue;
                }
                return Err(inst_err(
                    inst,
                    format!(
                        "Load missing resident copy for tensor {} on {} (refuse invent)",
                        tid.as_str(),
                        dest
                    ),
                ));
            }
        }
        Opcode::Transfer => {
            let src = inst.source.as_ref().map(|s| s.as_str()).unwrap_or("");
            let dst = inst
                .destination
                .as_ref()
                .map(|s| s.as_str())
                .unwrap_or(inst.resource.as_str());
            if src.is_empty() {
                return Err(inst_err(inst, "transfer missing source resource".into()));
            }
            *simulated = dst.contains("mock") || src.contains("mock");
            let mut sync_batch: Vec<(String, String, String, u64)> = Vec::new();
            for tid in inst.outputs.iter().chain(inst.inputs.iter()) {
                let tensor = TensorId::new(tid.as_str());
                if let Some(existing) =
                    residency.begin_transfer(&tensor, &ResourceId::new(dst), inst.name.as_str())
                {
                    // Share in-progress transfer — schedule deps must wait on producer.
                    let _ = existing;
                    continue;
                }
                // Strict: never invent a missing/stale source copy.
                residency.get(&tensor, &ResourceId::new(src)).map_err(|e| {
                    inst_err(
                        inst,
                        format!(
                            "transfer source copy missing or stale for tensor {} on {}: {}",
                            tid.as_str(),
                            src,
                            e
                        ),
                    )
                })?;
                // Event-derived liveness: lease source while transfer runs; lease
                // destination until Release (completion frontier).
                residency
                    .acquire_lease(&tensor, &ResourceId::new(src))
                    .map_err(|e| inst_err(inst, e.to_string()))?;
                let id = ctx.next_alloc_id();
                let replicate_result = residency.replicate(&tensor, ResourceId::new(dst), id, None);
                let _ = residency.release_lease(&tensor, &ResourceId::new(src));
                replicate_result.map_err(|e| inst_err(inst, e.to_string()))?;
                residency
                    .acquire_lease(&tensor, &ResourceId::new(dst))
                    .map_err(|e| inst_err(inst, e.to_string()))?;
                residency.end_transfer(&tensor, &ResourceId::new(dst));
                sync_batch.push((
                    tid.as_str().to_owned(),
                    src.to_owned(),
                    dst.to_owned(),
                    nbytes(inst, tid.as_str()),
                ));
            }
            if let Some(csync) = options.copy_sync.as_ref() {
                if !sync_batch.is_empty() {
                    csync(&sync_batch).map_err(|e| inst_err(inst, e))?;
                }
            }
            let xfer_bytes = inst.nbytes.max(1);
            if dst.contains("mock")
                || src.contains("mock")
                || attr_delay_s(inst, "mock_transfer_delay_s").is_some()
            {
                *simulated = true;
                simulate_mock_transfer(inst, ctx, dst, xfer_bytes)?;
            }
        }
        Opcode::RecordEvent => {
            ctx.completed_events().lock().insert(inst.name.to_string());
        }
        Opcode::WaitEvent => {
            let waits = inst
                .attr_str("waits_for")
                .map(str::to_owned)
                .or_else(|| inst.depends_on.first().map(|d| d.as_str().to_owned()));
            let Some(wf) = waits else {
                return Err(inst_err(
                    inst,
                    "WaitEvent missing waits_for / depends_on".into(),
                ));
            };
            // Strict: never invent a completed event. Dependency counters should
            // guarantee RecordEvent ran; if the event table lacks it, fail closed.
            if !ctx.completed_events().lock().contains(&wf) {
                return Err(inst_err(
                    inst,
                    format!("WaitEvent target {wf:?} was never recorded"),
                ));
            }
        }
        Opcode::Evict | Opcode::Release => {
            let kind = inst.attr_str("kind").unwrap_or("");
            if kind == "activation_spill" {
                return native_activation_spill(inst, ctx, residency, options);
            }
            let res = inst
                .attr_str("release_resource")
                .unwrap_or(inst.resource.as_str());
            let idempotent = matches!(
                inst.attributes.get("idempotent"),
                Some(streamcompiler_core::AttrValue::Bool(true))
            );
            let mut released_names: Vec<String> = Vec::new();
            for tid in &inst.inputs {
                let tensor = TensorId::new(tid.as_str());
                let mut resource = ResourceId::new(res);
                // After activation spill, the live copy may only remain on disk.
                if residency.get(&tensor, &resource).is_err()
                    && residency.get(&tensor, &ResourceId::new("disk")).is_ok()
                {
                    resource = ResourceId::new("disk");
                }
                // Drop event-derived leases before freeing the copy.
                let _ = residency.release_lease(&tensor, &resource);
                let alloc_id = residency
                    .get(&tensor, &resource)
                    .ok()
                    .map(|c| c.allocation.as_str().to_owned());
                match residency.release_copy(&tensor, &resource) {
                    Ok(freed) => {
                        if let Some(aid) = alloc_id.as_deref() {
                            ctx.free_virtual_buffer_for_alloc(aid, freed);
                        }
                        released_names.push(tid.as_str().to_owned());
                    }
                    Err(e) => {
                        let msg = e.to_string().to_lowercase();
                        if msg.contains("lease") || msg.contains("stale") || msg.contains("active")
                        {
                            return Err(inst_err(inst, e.to_string()));
                        }
                        // Missing target: only tolerate when explicitly marked idempotent.
                        if !idempotent {
                            return Err(inst_err(
                                inst,
                                format!(
                                    "release/evict target missing for tensor {} on {}: {}",
                                    tid.as_str(),
                                    res,
                                    e
                                ),
                            ));
                        }
                    }
                }
            }
            // Drop opaque Python handles in one callback (batched GIL cross).
            if let Some(href) = options.handle_release.as_ref() {
                if !released_names.is_empty() {
                    let pairs: Vec<(String, String)> = released_names
                        .into_iter()
                        .map(|tid| (tid, res.to_owned()))
                        .collect();
                    href(&pairs).map_err(|e| inst_err(inst, e))?;
                }
            }
        }
    }
    Ok(())
}

/// Release / parameter_evict ops that only need residency + handle-drop (no spill I/O).
fn is_batched_handle_release(inst: &streamcompiler_core::Instruction) -> bool {
    match inst.opcode {
        Opcode::Release => true,
        Opcode::Evict => {
            let kind = inst.attr_str("kind").unwrap_or("");
            kind != "activation_spill"
        }
        _ => false,
    }
}

fn native_activation_spill(
    inst: &streamcompiler_core::Instruction,
    ctx: &NativeExecutionContext,
    residency: &streamcompiler_memory::ResidencyStore,
    options: &ExecuteOptions,
) -> RuntimeResult<()> {
    let Some(demat) = options.dematerialize.as_ref() else {
        return Err(inst_err(
            inst,
            "activation_spill requires dematerialize callback; refuse silent RAM drop".into(),
        ));
    };
    let res = inst
        .attr_str("spill_resource")
        .or_else(|| inst.attr_str("release_resource"))
        .unwrap_or(inst.resource.as_str());
    for tid in &inst.inputs {
        let (meta, bytes) = demat(tid.as_str()).map_err(|e| inst_err(inst, e))?;
        let path = {
            let dir =
                ctx.with_storage(|st| st.spill_dir.clone().unwrap_or_else(std::env::temp_dir));
            streamcompiler_storage::write_activation_spill(&dir, &meta, &bytes)
                .map_err(|e| inst_err(inst, e.to_string()))?
        };
        ctx.with_storage(|st| {
            st.spills.insert(tid.as_str().to_owned(), path.clone());
            st.bytes_written += meta.nbytes;
        });
        let tensor = TensorId::new(tid.as_str());
        let resource = ResourceId::new(res);
        let _ = residency.release_lease(&tensor, &resource);
        let alloc_id = residency
            .get(&tensor, &resource)
            .ok()
            .map(|c| c.allocation.as_str().to_owned());
        // Spill already wrote disk bytes — failing to drop the RAM copy is a leak / bug.
        let freed = residency.release_copy(&tensor, &resource).map_err(|e| {
            inst_err(
                inst,
                format!(
                    "activation_spill failed to release {} on {}: {}",
                    tid.as_str(),
                    res,
                    e
                ),
            )
        })?;
        if let Some(aid) = alloc_id.as_deref() {
            ctx.free_virtual_buffer_for_alloc(aid, freed);
        }
        let id = ctx.next_alloc_id();
        residency
            .put(
                tensor,
                ResourceId::new("disk"),
                id,
                TensorMetadata {
                    nbytes: meta.nbytes,
                    dtype: meta.dtype,
                    shape: meta.shape,
                    ..Default::default()
                },
                None,
            )
            .map_err(|e| inst_err(inst, e.to_string()))?;
        let _ = path;
    }
    Ok(())
}

fn native_activation_reload(
    inst: &streamcompiler_core::Instruction,
    ctx: &NativeExecutionContext,
    residency: &streamcompiler_memory::ResidencyStore,
    options: &ExecuteOptions,
) -> RuntimeResult<()> {
    let Some(mat) = options.materialize.as_ref() else {
        return Err(inst_err(
            inst,
            "activation_reload requires materialize callback".into(),
        ));
    };
    let dest = inst
        .destination
        .as_ref()
        .map(|d| d.as_str())
        .unwrap_or(inst.resource.as_str());
    for tid in &inst.inputs {
        let path = ctx
            .with_storage(|st| st.spills.get(tid.as_str()).cloned())
            .ok_or_else(|| {
                inst_err(
                    inst,
                    format!("activation_reload missing spill file for {}", tid.as_str()),
                )
            })?;
        let (meta, bytes) = streamcompiler_storage::read_activation_spill(&path)
            .map_err(|e| inst_err(inst, e.to_string()))?;
        mat(tid.as_str(), &meta, &bytes).map_err(|e| inst_err(inst, e))?;
        ctx.with_storage(|st| {
            st.bytes_read += meta.nbytes;
            st.spills.remove(tid.as_str());
        });
        let _ = streamcompiler_storage::remove_activation_spill(&path);
        let tensor = TensorId::new(tid.as_str());
        if residency.get(&tensor, &ResourceId::new(dest)).is_err() {
            let id = ctx.next_alloc_id();
            residency
                .put(
                    tensor,
                    ResourceId::new(dest),
                    id,
                    TensorMetadata {
                        nbytes: meta.nbytes,
                        dtype: meta.dtype.clone(),
                        shape: meta.shape.clone(),
                        ..Default::default()
                    },
                    None,
                )
                .map_err(|e| inst_err(inst, e.to_string()))?;
        }
    }
    Ok(())
}

fn nbytes(inst: &streamcompiler_core::Instruction, tensor: &str) -> u64 {
    inst.tensor_nbytes()
        .get(tensor)
        .copied()
        .unwrap_or(inst.nbytes)
        .max(1)
}

fn load_notes(
    inst: &streamcompiler_core::Instruction,
    residency: &streamcompiler_memory::ResidencyStore,
) -> String {
    if inst.opcode != Opcode::Load {
        return "native_data_plane".into();
    }
    let dest = inst
        .destination
        .as_ref()
        .map(|d| d.as_str())
        .unwrap_or(inst.resource.as_str());
    let all_resident = inst.outputs.iter().chain(inst.inputs.iter()).all(|tid| {
        residency
            .get(&TensorId::new(tid.as_str()), &ResourceId::new(dest))
            .is_ok()
    });
    if all_resident {
        "persistent_residency".into()
    } else {
        "native_data_plane".into()
    }
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
        let report =
            execute_schedule(&schedule, &ExecuteOptions::default(), Some(cb), None).unwrap();
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
        let report =
            execute_schedule_with_context(&schedule, &opts, Some(region), None, ctx).unwrap();
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
            stream_id: Some(streamcompiler_core::StreamId::new("cpu::copy0")),
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
            stream_id: Some(streamcompiler_core::StreamId::new("cpu::compute0")),
            copy_engine_id: None,
            link_id: None,
            io_queue_id: None,
            attributes: {
                let mut m = IndexMap::new();
                m.insert(
                    "waits_for".into(),
                    streamcompiler_core::AttrValue::String("never_recorded".into()),
                );
                m
            },
        };
        let ctx = NativeExecutionContext::new();
        let err =
            run_instruction(&wait, &ctx, None, false, &ExecuteOptions::default()).unwrap_err();
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
            stream_id: Some(streamcompiler_core::StreamId::new("cpu::io0")),
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
            stream_id: Some(streamcompiler_core::StreamId::new("cpu::compute0")),
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
            streamcompiler_core::AttrValue::String("activation_spill".into()),
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
        let err =
            run_instruction(&spill, &ctx, None, false, &ExecuteOptions::default()).unwrap_err();
        assert!(
            err.to_string().contains("dematerialize")
                || err.to_string().contains("activation_spill"),
            "unexpected: {err}"
        );
        assert!(
            ctx.residency()
                .get(&TensorId::new("act"), &ResourceId::new("cpu"))
                .is_ok(),
            "spill must not drop RAM when body missing"
        );
    }
}
