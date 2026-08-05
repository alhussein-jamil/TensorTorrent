//! Dependency-counter schedule executor with resource queues and compute callbacks.

mod capacity;
mod instruction;
mod pooled;
mod spill;
mod waves;

#[cfg(test)]
mod tests;

use crate::context::NativeExecutionContext;
use crate::error::{RuntimeError, RuntimeResult};
use crate::telemetry::InstructionTelemetry;
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tt_ir::{assert_schedule_valid, ExecutableSchedule, Opcode};
use tt_ir::{ResourceId, TensorId};
use tt_memory::TensorMetadata;

use instruction::{inst_err, load_notes, nbytes, run_instruction, simulate_mock_compute};
use pooled::execute_schedule_pooled;
use waves::{
    is_batched_handle_release, is_batched_parameter_load, is_native_launch,
    run_parameter_load_wave, run_release_wave,
};
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
    Arc<dyn Fn(&str) -> Result<(tt_storage::SpillMeta, Vec<u8>), String> + Send + Sync>;

/// Contiguous host bytes → tensor registration for native activation reload.
pub type MaterializeCallback =
    Arc<dyn Fn(&str, &tt_storage::SpillMeta, &[u8]) -> Result<(), String> + Send + Sync>;

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

/// Pre-computed dependency graph over ``schedule.instructions``.
///
/// State is index-based (0..instructions.len()); building the graph does zero
/// String allocation. Shared helper used by every schedule executor path.
pub(crate) struct DepGraph {
    /// `remaining[i]` = unmet dependency count for instruction index `i`.
    remaining: Vec<usize>,
    /// `dependents[i]` = indices of instructions that depend on instruction `i`.
    dependents: Vec<Vec<usize>>,
    /// Initial ready set: instructions with zero unmet dependencies.
    initial_ready: Vec<usize>,
}

impl DepGraph {
    fn build(schedule: &ExecutableSchedule) -> Self {
        let n = schedule.instructions.len();
        // Only used during construction to resolve depends_on names → indices.
        let mut name_to_index: HashMap<&str, usize> = HashMap::with_capacity(n);
        for (idx, inst) in schedule.instructions.iter().enumerate() {
            name_to_index.insert(inst.name.as_str(), idx);
        }
        let remaining: Vec<usize> = schedule
            .instructions
            .iter()
            .map(|i| i.depends_on.len())
            .collect();
        let mut dependents: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (idx, inst) in schedule.instructions.iter().enumerate() {
            for dep in &inst.depends_on {
                if let Some(&dep_idx) = name_to_index.get(dep.as_str()) {
                    dependents[dep_idx].push(idx);
                }
            }
        }
        let initial_ready: Vec<usize> = (0..n).filter(|&i| remaining[i] == 0).collect();
        Self {
            remaining,
            dependents,
            initial_ready,
        }
    }
}

pub(crate) enum WorkKind {
    Cpu,
    Io,
    Transfer,
    Inline,
}

pub(crate) struct Completion {
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

pub(crate) fn max_ready_width(schedule: &ExecutableSchedule) -> usize {
    let graph = DepGraph::build(schedule);
    let mut remaining = graph.remaining;
    let mut ready: VecDeque<usize> = graph.initial_ready.into_iter().collect();
    let mut peak = ready.len();
    while let Some(idx) = ready.pop_front() {
        for &nxt in &graph.dependents[idx] {
            let deg = &mut remaining[nxt];
            *deg = deg.saturating_sub(1);
            if *deg == 0 {
                ready.push_back(nxt);
            }
        }
        peak = peak.max(ready.len());
    }
    peak
}

pub(crate) fn needs_python_io(inst: &tt_ir::Instruction, opts: &ExecuteOptions) -> bool {
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
            // Fall back to Python I/O when native materialize/load hooks are absent.
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

    let graph = DepGraph::build(schedule);
    let mut remaining = graph.remaining;
    let mut ready: VecDeque<usize> = graph.initial_ready.into_iter().collect();
    let dependents = &graph.dependents;
    let inst_at = |idx: usize| -> &tt_ir::Instruction { &schedule.instructions[idx] };

    let mut events = Vec::with_capacity(schedule.instructions.len());
    let mut bytes_read = 0u64;
    let mut bytes_transferred = 0u64;
    let mut simulated_ops = 0usize;
    let origin = t0;

    // Reusable per-wave scratch buffers avoid VecDeque/Vec reallocation churn.
    let mut compute_indices: Vec<usize> = Vec::new();
    let mut other: VecDeque<usize> = VecDeque::new();
    let mut rest: VecDeque<usize> = VecDeque::new();

    // Reusable release/load wave scratch.
    let mut release_indices: Vec<usize> = Vec::new();
    let mut release_insts: Vec<&tt_ir::Instruction> = Vec::new();
    let mut load_indices: Vec<usize> = Vec::new();
    let mut load_insts: Vec<&tt_ir::Instruction> = Vec::new();

    // Push a completed instruction's dependents back onto the ready queue.
    macro_rules! release_deps {
        ($idx:expr) => {
            for &nxt in &dependents[$idx] {
                let deg = &mut remaining[nxt];
                *deg = deg.saturating_sub(1);
                if *deg == 0 {
                    ready.push_back(nxt);
                }
            }
        };
    }

    while !ready.is_empty() {
        if cancel.load(Ordering::Acquire) {
            return Err(Box::new(RuntimeError::Cancelled));
        }
        compute_indices.clear();
        other.clear();
        while let Some(idx) = ready.pop_front() {
            let inst = inst_at(idx);
            if inst.opcode == Opcode::Compute {
                compute_indices.push(idx);
            } else {
                other.push_back(idx);
            }
        }
        // Preserve original schedule order for Compute waves (deterministic dispatch).
        compute_indices.sort_unstable();

        // Wave-batch ready Releases / parameter_evicts before other I/O (one GIL).
        if options.handle_release.is_some() {
            release_indices.clear();
            release_insts.clear();
            rest.clear();
            while let Some(idx) = other.pop_front() {
                let inst = inst_at(idx);
                if is_batched_handle_release(inst) {
                    release_indices.push(idx);
                    release_insts.push(inst);
                } else {
                    rest.push_back(idx);
                }
            }
            std::mem::swap(&mut other, &mut rest);
            if !release_insts.is_empty() {
                let teles = run_release_wave(&release_insts, ctx, options, origin)?;
                for (tel, &idx) in teles.into_iter().zip(release_indices.iter()) {
                    if matches!(tel.opcode.as_str(), "Load" | "Prefetch") {
                        bytes_read += tel.nbytes;
                    }
                    if tel.simulated {
                        simulated_ops += 1;
                    }
                    events.push(tel);
                    release_deps!(idx);
                }
            }
        }

        // Wave-batch ready parameter Loads (one GIL for the whole wave).
        if options.parameter_load.is_some() {
            load_indices.clear();
            load_insts.clear();
            rest.clear();
            while let Some(idx) = other.pop_front() {
                let inst = inst_at(idx);
                if is_batched_parameter_load(inst) {
                    load_indices.push(idx);
                    load_insts.push(inst);
                } else {
                    rest.push_back(idx);
                }
            }
            std::mem::swap(&mut other, &mut rest);
            if !load_insts.is_empty() {
                let teles = run_parameter_load_wave(&load_insts, ctx, options, origin)?;
                for (tel, &idx) in teles.into_iter().zip(load_indices.iter()) {
                    bytes_read += tel.nbytes;
                    events.push(tel);
                    release_deps!(idx);
                }
            }
        }

        // Prefetch/Transfer/Load first so background I/O overlaps the Compute wave.
        while let Some(idx) = other.pop_front() {
            if cancel.load(Ordering::Acquire) {
                return Err(Box::new(RuntimeError::Cancelled));
            }
            let inst = inst_at(idx);
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
                name: inst.name.as_str().to_owned(),
                opcode: inst.opcode.to_string(),
                resource: inst.resource.to_string(),
                submitted_s: submitted,
                start_s: start,
                end_s: end,
                nbytes: nbytes_out,
                simulated,
                notes,
            });
            release_deps!(idx);
        }

        if !compute_indices.is_empty() {
            // Partition into native-launch (Rust-only) and Python-region groups.
            let mut native_indices: Vec<usize> = Vec::new();
            let mut py_indices: Vec<usize> = Vec::new();
            for &idx in &compute_indices {
                if is_native_launch(inst_at(idx)) {
                    native_indices.push(idx);
                } else {
                    py_indices.push(idx);
                }
            }
            compute_indices.clear();

            for idx in native_indices {
                let inst = inst_at(idx);
                let submitted = origin.elapsed().as_secs_f64();
                let start = submitted;
                let simulated = run_instruction(inst, ctx, None, false, options)?;
                let end = origin.elapsed().as_secs_f64();
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
                    notes: "native_launch".into(),
                });
                release_deps!(idx);
            }
            if py_indices.is_empty() {
                continue;
            }
            let mut invocations: Vec<RegionInvocation> = Vec::with_capacity(py_indices.len());
            let mut batch_meta: Vec<(&tt_ir::Instruction, f64, f64)> =
                Vec::with_capacity(py_indices.len());
            for &idx in &py_indices {
                let inst = inst_at(idx);
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
                batch_meta.push((inst, submitted, submitted));
            }
            region_cb(&invocations).map_err(|cause| {
                let region = invocations
                    .first()
                    .map(|i| i.region_id.clone())
                    .unwrap_or_default();
                Box::new(RuntimeError::Instruction {
                    instruction: py_indices
                        .first()
                        .map(|&i| inst_at(i).name.as_str().to_owned())
                        .unwrap_or_else(|| "compute_batch".into()),
                    opcode: "Compute".into(),
                    region: Some(region),
                    tensor: None,
                    resource: None,
                    cause,
                })
            })?;
            let batch_note = if py_indices.len() > 1 {
                format!("region_callback_batch:{}", py_indices.len())
            } else {
                "region_callback".into()
            };
            for (i, (inst, submitted, start)) in batch_meta.into_iter().enumerate() {
                let idx = py_indices[i];
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
                release_deps!(idx);
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
    let graph = DepGraph::build(schedule);
    let mut remaining = graph.remaining;
    let mut ready: VecDeque<usize> = graph.initial_ready.into_iter().collect();

    let mut events = Vec::with_capacity(schedule.instructions.len());
    let mut bytes_read = 0u64;
    let mut bytes_transferred = 0u64;
    let mut simulated_ops = 0usize;
    let origin = t0;

    while let Some(idx) = ready.pop_front() {
        if cancel.load(Ordering::Acquire) {
            return Err(Box::new(RuntimeError::Cancelled));
        }
        let inst = &schedule.instructions[idx];
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
            name: inst.name.as_str().to_owned(),
            opcode: inst.opcode.to_string(),
            resource: inst.resource.to_string(),
            submitted_s: submitted,
            start_s: start,
            end_s: end,
            nbytes: outcome.nbytes,
            simulated: outcome.simulated,
            notes: outcome.notes,
        });
        for &nxt in &graph.dependents[idx] {
            let deg = &mut remaining[nxt];
            *deg = deg.saturating_sub(1);
            if *deg == 0 {
                ready.push_back(nxt);
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
fn execute_dry_run_inline(
    schedule: &ExecutableSchedule,
    ctx: &NativeExecutionContext,
    t0: Instant,
) -> RuntimeResult<ExecuteReport> {
    let graph = DepGraph::build(schedule);
    let mut remaining = graph.remaining;
    let mut ready: VecDeque<usize> = graph.initial_ready.into_iter().collect();
    let mut events = Vec::with_capacity(schedule.instructions.len());
    while let Some(idx) = ready.pop_front() {
        let inst = &schedule.instructions[idx];
        let start = t0.elapsed().as_secs_f64();
        let end = start;
        events.push(InstructionTelemetry {
            name: inst.name.as_str().to_owned(),
            opcode: inst.opcode.to_string(),
            resource: inst.resource.to_string(),
            submitted_s: start,
            start_s: start,
            end_s: end,
            nbytes: inst.nbytes,
            simulated: false,
            notes: "dry_run".into(),
        });
        for &nxt in &graph.dependents[idx] {
            let deg = &mut remaining[nxt];
            *deg = deg.saturating_sub(1);
            if *deg == 0 {
                ready.push_back(nxt);
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
