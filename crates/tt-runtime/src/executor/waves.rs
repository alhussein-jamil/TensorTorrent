//! Dependency-counter schedule executor with resource queues and compute callbacks.

use crate::context::NativeExecutionContext;
use crate::error::{RuntimeError, RuntimeResult};
use crate::telemetry::InstructionTelemetry;
use parking_lot::Mutex;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::Instant;
use tt_ir::Opcode;
use tt_ir::{ResourceId, TensorId};

use super::capacity::order_key;
use super::instruction::run_instruction;
use super::{ExecuteOptions, HandleReleaseCallback};
pub(crate) fn enqueue_ready(
    name: &str,
    by_name: &HashMap<&str, &tt_ir::Instruction>,
    ready: &mut VecDeque<String>,
    resource_queues: &mut HashMap<String, VecDeque<String>>,
    resource_busy: &HashSet<String>,
) {
    let Some(&inst) = by_name.get(name) else {
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

/// Release / parameter_evict ops that only need residency + handle-drop (no spill I/O).
pub(crate) fn is_batched_handle_release(inst: &tt_ir::Instruction) -> bool {
    match inst.opcode {
        Opcode::Release => true,
        Opcode::Evict => {
            let kind = inst.attr_str("kind").unwrap_or("");
            kind != "activation_spill"
        }
        _ => false,
    }
}

/// Pull ready Release / parameter_evict ops; leave everything else in `ready`.
pub(crate) fn take_ready_release_wave(
    ready: &mut VecDeque<String>,
    by_name: &HashMap<&str, &tt_ir::Instruction>,
) -> Vec<String> {
    let mut wave = Vec::new();
    let mut rest = VecDeque::new();
    while let Some(name) = ready.pop_front() {
        let Some(&inst) = by_name.get(name.as_str()) else {
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
pub(crate) fn take_ready_parameter_load_wave(
    ready: &mut VecDeque<String>,
    by_name: &HashMap<&str, &tt_ir::Instruction>,
) -> Vec<String> {
    let mut wave = Vec::new();
    let mut rest = VecDeque::new();
    while let Some(name) = ready.pop_front() {
        let Some(&inst) = by_name.get(name.as_str()) else {
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

pub(crate) fn is_batched_parameter_load(inst: &tt_ir::Instruction) -> bool {
    inst.opcode == Opcode::Load && inst.attr_str("kind").unwrap_or("") == "parameter_materialize"
}

/// Run a parameter-load wave: residency checks per-op, one batched callback.
pub(crate) fn run_parameter_load_wave(
    wave_insts: &[&tt_ir::Instruction],
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
pub(crate) fn run_release_wave(
    wave_insts: &[&tt_ir::Instruction],
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
    let pairs: Vec<(String, String)> = std::mem::take(&mut *collected.lock());
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

pub(crate) fn is_native_launch(inst: &tt_ir::Instruction) -> bool {
    inst.attr_bool("native_launch").unwrap_or(false)
}

/// Drain ready Computes that are not stream/engine-blocked into one wave.
/// Marks their order keys busy. Returns `None` when no Compute is launchable.
pub(crate) fn take_ready_compute_wave(
    ready: &mut VecDeque<String>,
    by_name: &HashMap<&str, &tt_ir::Instruction>,
    resource_busy: &mut HashSet<String>,
    resource_queues: &mut HashMap<String, VecDeque<String>>,
) -> Option<Vec<String>> {
    let mut wave = Vec::new();
    let mut deferred = VecDeque::new();
    while let Some(name) = ready.pop_front() {
        let Some(&inst) = by_name.get(name.as_str()) else {
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
