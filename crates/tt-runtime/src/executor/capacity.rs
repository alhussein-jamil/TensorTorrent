//! Dependency-counter schedule executor with resource queues and compute callbacks.

use crate::context::NativeExecutionContext;
use crate::error::{RuntimeError, RuntimeResult};
use tt_ir::Opcode;

use super::WorkKind;
/// Serialize on stream_id when present; never whole-device when streams differ.
pub(crate) fn order_key(inst: &tt_ir::Instruction) -> Option<String> {
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

pub(crate) fn is_ordered_resource(res: &str) -> bool {
    let l = res.to_lowercase();
    l.contains("mock")
        || l.contains("cuda")
        || l.contains("rocm")
        || l.contains("gpu")
        || l.contains("accel")
}

pub(crate) fn work_kind(op: Opcode) -> WorkKind {
    match op {
        Opcode::Compute => WorkKind::Cpu,
        Opcode::Prefetch | Opcode::Load | Opcode::Evict => WorkKind::Io,
        Opcode::Transfer => WorkKind::Transfer,
        Opcode::RecordEvent | Opcode::WaitEvent | Opcode::Release => WorkKind::Inline,
    }
}

/// Wait for a bounded resource slot without spinning a core.
///
/// The previous implementation busy-waited on `yield_now()` forever: a lost
/// completion pinned a CPU at 100% and hung the forward. This waits with a
/// short sleep and a progress-generation watchdog — the deadline only fires
/// when NOTHING in the execution has made progress for the stall timeout,
/// so legitimately slow transfers never trip it.
pub(crate) fn acquire_capacity(
    ctx: &NativeExecutionContext,
    kind: &str,
    id: &str,
    max_concurrent: u32,
) -> RuntimeResult<()> {
    wait_for_resource(ctx, &format!("{kind} capacity on {id}"), || {
        ctx.with_resources(|rs| {
            let cap = match kind {
                "copy" => rs.ensure_copy_engine(id, max_concurrent),
                "io" => rs.ensure_io_queue(id, max_concurrent),
                _ => return true,
            };
            cap.try_acquire()
        })
    })
}

/// Shared stall-aware wait loop for resource acquisition.
pub(crate) fn wait_for_resource(
    ctx: &NativeExecutionContext,
    what: &str,
    mut try_acquire: impl FnMut() -> bool,
) -> RuntimeResult<()> {
    let timeout = ctx.stall_timeout();
    let started = std::time::Instant::now();
    let mut last_gen = ctx.progress_generation();
    let mut last_progress = std::time::Instant::now();
    loop {
        if ctx.is_cancelled() {
            return Err(Box::new(RuntimeError::Cancelled));
        }
        if try_acquire() {
            return Ok(());
        }
        let gen = ctx.progress_generation();
        if gen != last_gen {
            last_gen = gen;
            last_progress = std::time::Instant::now();
        } else if let Some(limit) = timeout {
            if last_progress.elapsed() > limit {
                return Err(Box::new(RuntimeError::Stalled {
                    what: what.to_owned(),
                    waited_s: started.elapsed().as_secs_f64(),
                }));
            }
        }
        std::thread::sleep(std::time::Duration::from_micros(200));
    }
}

pub(crate) fn acquire_link(
    ctx: &NativeExecutionContext,
    link_id: &str,
    nbytes: u64,
) -> RuntimeResult<()> {
    wait_for_resource(ctx, &format!("link {link_id}"), || {
        ctx.with_resources(|rs| {
            let link = rs.links.entry(link_id.to_owned()).or_default();
            // One transfer per link at a time (contention).
            if link.bytes_in_flight > 0 {
                return false;
            }
            link.bytes_in_flight = nbytes.max(1);
            true
        })
    })
}
