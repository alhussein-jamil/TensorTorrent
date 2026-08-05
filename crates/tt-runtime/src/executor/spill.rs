//! Dependency-counter schedule executor with resource queues and compute callbacks.

use crate::context::NativeExecutionContext;
use crate::error::RuntimeResult;
use tt_ir::{ResourceId, TensorId};
use tt_memory::TensorMetadata;

use super::instruction::inst_err;
use super::ExecuteOptions;
pub(crate) fn native_activation_spill(
    inst: &tt_ir::Instruction,
    ctx: &NativeExecutionContext,
    residency: &tt_memory::ResidencyStore,
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
        // Aggregate spill budget: the per-file cap alone cannot stop N
        // concurrent spills from filling the disk.
        ctx.with_storage(|st| -> Result<(), String> {
            if let Some(budget) = st.spill_budget_bytes {
                let projected = st.spill_live_bytes.saturating_add(meta.nbytes);
                if projected > budget {
                    return Err(format!(
                        "spill budget exceeded: live={} + requested={} > budget={} bytes                          (raise CompileConfig.max_total_spill_bytes or lower                          activation pressure)",
                        st.spill_live_bytes, meta.nbytes, budget
                    ));
                }
            }
            Ok(())
        })
        .map_err(|e| inst_err(inst, e))?;
        let path = {
            let dir = ctx.spill_session_dir().map_err(|e| inst_err(inst, e))?;
            tt_storage::write_activation_spill(&dir, &meta, &bytes)
                .map_err(|e| inst_err(inst, e.to_string()))?
        };
        ctx.with_storage(|st| {
            st.spills.insert(tid.as_str().to_owned(), path.clone());
            st.bytes_written += meta.nbytes;
            st.spill_live_bytes = st.spill_live_bytes.saturating_add(meta.nbytes);
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

pub(crate) fn native_activation_reload(
    inst: &tt_ir::Instruction,
    ctx: &NativeExecutionContext,
    residency: &tt_memory::ResidencyStore,
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
        let (meta, bytes) =
            tt_storage::read_activation_spill(&path).map_err(|e| inst_err(inst, e.to_string()))?;
        mat(tid.as_str(), &meta, &bytes).map_err(|e| inst_err(inst, e))?;
        ctx.with_storage(|st| {
            st.bytes_read += meta.nbytes;
            st.spills.remove(tid.as_str());
            st.spill_live_bytes = st.spill_live_bytes.saturating_sub(meta.nbytes);
        });
        let _ = tt_storage::remove_activation_spill(&path);
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
