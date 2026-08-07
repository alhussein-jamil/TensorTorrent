//! Dependency-counter schedule executor with resource queues and compute callbacks.

use crate::context::NativeExecutionContext;
use crate::error::{RuntimeError, RuntimeResult};
use tt_ir::Opcode;
use tt_ir::{ResourceId, TensorId};
use tt_memory::TensorMetadata;

use super::capacity::{acquire_capacity, acquire_link};
use super::spill::{native_activation_reload, native_activation_spill};
use super::{ExecuteOptions, RegionCallback, RegionInvocation};
pub(crate) fn run_instruction(
    inst: &tt_ir::Instruction,
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
        ctx.bump_progress();
    }
    if let Some(ref qid) = inst.io_queue_id {
        ctx.with_resources(|rs| {
            if let Some(q) = rs.io_queues.get_mut(qid.as_str()) {
                q.release();
            }
        });
        ctx.bump_progress();
    }
    if let Some(ref eng) = inst.copy_engine_id {
        ctx.with_resources(|rs| {
            if let Some(cap) = rs.copy_engines.get_mut(eng.as_str()) {
                cap.release();
            }
        });
        ctx.bump_progress();
    }
    if let Some(ref sid) = inst.stream_id {
        ctx.with_resources(|rs| rs.note_stream_complete(sid.as_str()));
    }
    result?;
    Ok(simulated)
}

pub(crate) fn attr_delay_s(inst: &tt_ir::Instruction, key: &str) -> Option<f64> {
    inst.attributes.get(key).and_then(|v| match v {
        tt_ir::AttrValue::Float(f) => Some(*f),
        tt_ir::AttrValue::Int(i) => Some(*i as f64),
        _ => None,
    })
}

/// Public mock path: Rust virtual backend (buffers/streams/pending events), not Python MockStream.
pub(crate) fn simulate_mock_compute(
    inst: &tt_ir::Instruction,
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

pub(crate) fn simulate_mock_transfer(
    inst: &tt_ir::Instruction,
    ctx: &NativeExecutionContext,
    dst: &str,
    nbytes: u64,
) -> RuntimeResult<()> {
    let delay = attr_delay_s(inst, "mock_transfer_delay_s");
    // Real H2D already ran in copy_sync. Virtual-backend transfer only for mock
    // resources or an explicit positive delay — not Some(0.0).
    let positive_delay = delay.map(|d| d > 0.0).unwrap_or(false);
    if !positive_delay && !dst.contains("mock") && !inst.resource.as_str().contains("mock") {
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
    let transfer_bytes = usize::try_from(nbytes.max(1)).map_err(|_| {
        inst_err(
            inst,
            format!("virtual transfer size {nbytes} does not fit usize"),
        )
    })?;
    let be = ctx.virtual_backend(resource);
    be.run_transfer(&stream, transfer_bytes, delay)
        .map_err(|e| inst_err(inst, format!("virtual backend transfer (simulated): {e}")))
}
pub(crate) fn run_instruction_body(
    inst: &tt_ir::Instruction,
    ctx: &NativeExecutionContext,
    residency: &tt_memory::ResidencyStore,
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
            let native_launch = inst.attr_bool("native_launch").unwrap_or(false);
            if native_launch {
                // AOT / backend launch path: no Python region callback (GIL-free).
                // Torch regions still use region_cb until Inductor/AOT artifacts land.
                let delay = attr_delay_s(inst, "mock_compute_delay_s").unwrap_or(0.0);
                let stream = inst
                    .stream_id
                    .as_ref()
                    .map(|s| s.as_str().to_owned())
                    .unwrap_or_else(|| format!("{}::compute0", inst.resource.as_str()));
                let be = ctx.virtual_backend(inst.resource.as_str());
                be.run_compute(&stream, delay)
                    .map_err(|e| inst_err(inst, format!("native_launch: {e}")))?;
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
                }
            } else if dry_run || region_cb.is_none() {
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
            let mock_delay = attr_delay_s(inst, "mock_transfer_delay_s").unwrap_or(0.0);
            if dst.contains("mock") || src.contains("mock") || mock_delay > 0.0 {
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
                Some(tt_ir::AttrValue::Bool(true))
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
pub(crate) fn nbytes(inst: &tt_ir::Instruction, tensor: &str) -> u64 {
    inst.tensor_nbytes()
        .get(tensor)
        .copied()
        .unwrap_or(inst.nbytes)
        .max(1)
}

pub(crate) fn load_notes(
    inst: &tt_ir::Instruction,
    residency: &tt_memory::ResidencyStore,
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

pub(crate) fn inst_err(inst: &tt_ir::Instruction, cause: String) -> Box<RuntimeError> {
    Box::new(RuntimeError::Instruction {
        instruction: inst.name.to_string(),
        opcode: inst.opcode.to_string(),
        region: inst.executable_ref.as_ref().map(|r| r.to_string()),
        tensor: inst.inputs.first().map(|t| t.to_string()),
        resource: Some(inst.resource.to_string()),
        cause,
    })
}
