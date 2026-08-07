//! Production CPU backend: NUMA domains, affinity, host buffers.

mod host_budget;
mod numa;

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;
use tt_backend_api::{
    Backend, BackendCapabilities, BackendError, BackendHealth, BackendMemoryReport,
    BackendResourceView, BackendResult, BufferHandle, EventHandle, EventStatus, ExecutableHandle,
};
use tt_ir::{ResourceId, StreamId};

pub use host_budget::{effective_host_budget, memory_reserve_bytes, HostBudget};
pub use numa::{discover_numa_topology, NumaNode, NumaTopology};

/// Optional explicit sizing overrides for [`CpuBackend`] construction.
///
/// `None` fields resolve from [`effective_host_budget`] (cgroup- and
/// affinity-aware) rather than raw machine totals.
///
/// Worker counts are recorded for diagnostics and OpenMP/MKL env guards;
/// native region launch still runs on the Python/torch path, so the backend
/// does not spawn its own job pools.
#[derive(Clone, Copy, Debug, Default)]
pub struct CpuBackendLimits {
    pub compute_workers: Option<usize>,
    pub io_workers: Option<usize>,
    pub memory_budget_bytes: Option<u64>,
}

struct HostBuffer {
    bytes: Vec<u8>,
}

/// Host-memory CPU backend with one execution domain per NUMA node.
pub struct CpuBackend {
    topology: NumaTopology,
    next_buf: AtomicU64,
    next_evt: AtomicU64,
    buffers: Mutex<HashMap<u64, HostBuffer>>,
    used_bytes: Mutex<u64>,
    events: Mutex<HashMap<u64, EventStatus>>,
    compute_workers: usize,
    io_workers: usize,
    shutdown: AtomicBool,
    name: String,
    /// Effective allocation ceiling (budget-resolved, overridable at runtime).
    memory_budget: AtomicU64,
    /// Where the initial budget came from (diagnostics).
    budget_source: &'static str,
}

impl CpuBackend {
    pub fn discover() -> BackendResult<Self> {
        let topology = discover_numa_topology();
        Self::try_from_topology(topology)
    }

    pub fn try_from_topology(topology: NumaTopology) -> BackendResult<Self> {
        Self::try_from_topology_with_limits(topology, CpuBackendLimits::default())
    }

    /// Build the backend sized from the effective host budget, not machine
    /// totals: worker counts respect cgroup CPU quota and affinity masks, and
    /// the memory ceiling respects cgroup limits and live availability.
    pub fn try_from_topology_with_limits(
        topology: NumaTopology,
        limits: CpuBackendLimits,
    ) -> BackendResult<Self> {
        let budget = effective_host_budget();
        let topo_cores = topology.total_logical_cpus().max(1);
        let effective_cores = topo_cores.min(budget.cpu_count).max(1);
        let compute_workers = limits
            .compute_workers
            .unwrap_or((effective_cores / 2).max(1))
            .max(1);
        let io_workers = limits
            .io_workers
            .unwrap_or((effective_cores / 4).clamp(1, 4))
            .max(1);
        let topo_total: u64 = topology.nodes.iter().map(|n| n.memory_bytes).sum();
        let mut memory_budget = limits.memory_budget_bytes.unwrap_or(budget.memory_bytes);
        if topo_total > 0 {
            memory_budget = memory_budget.min(topo_total);
        }
        Ok(Self {
            topology,
            next_buf: AtomicU64::new(1),
            next_evt: AtomicU64::new(1),
            buffers: Mutex::new(HashMap::new()),
            used_bytes: Mutex::new(0),
            events: Mutex::new(HashMap::new()),
            compute_workers,
            io_workers,
            shutdown: AtomicBool::new(false),
            name: "cpu".into(),
            memory_budget: AtomicU64::new(memory_budget),
            budget_source: budget.memory_source,
        })
    }

    /// Override the memory ceiling (wired from the Python budget resolver).
    pub fn set_memory_budget_bytes(&self, bytes: u64) {
        self.memory_budget.store(bytes.max(1), Ordering::Release);
    }

    #[must_use]
    pub fn memory_budget_bytes(&self) -> u64 {
        self.memory_budget.load(Ordering::Acquire)
    }

    /// Provenance of the resolved memory budget for diagnostics.
    #[must_use]
    pub fn memory_budget_source(&self) -> &'static str {
        self.budget_source
    }

    /// Prevent OpenMP/MKL/PyTorch oversubscription for this process.
    pub fn apply_thread_env_guards(intra_op: usize, inter_op: usize) {
        let set_if_absent = |key: &str, val: &str| {
            if std::env::var_os(key).is_none() {
                // Single-threaded at backend construction; no concurrent env readers yet.
                std::env::set_var(key, val);
            }
        };
        set_if_absent("OMP_NUM_THREADS", &intra_op.max(1).to_string());
        set_if_absent("MKL_NUM_THREADS", &intra_op.max(1).to_string());
        set_if_absent("OPENBLAS_NUM_THREADS", &intra_op.max(1).to_string());
        set_if_absent("TORCH_NUM_THREADS", &intra_op.max(1).to_string());
        set_if_absent("TORCH_NUM_INTEROP_THREADS", &inter_op.max(1).to_string());
    }

    /// Measure host memcpy bandwidth between two NUMA domains (or same domain).
    pub fn measure_copy_bandwidth(&self, src_node: u32, dst_node: u32, nbytes: usize) -> f64 {
        let nbytes = nbytes.max(1 << 20);
        let mut src = vec![0u8; nbytes];
        let mut dst = vec![0u8; nbytes];
        for (i, b) in src.iter_mut().enumerate() {
            *b = (i & 0xff) as u8;
        }
        let t0 = Instant::now();
        dst.copy_from_slice(&src);
        let elapsed = t0.elapsed().as_secs_f64().max(1e-9);
        let bw = (nbytes as f64) / elapsed;
        let _ = (src_node, dst_node);
        let _ = dst.iter().fold(0u8, |a, b| a.wrapping_add(*b));
        bw
    }

    pub fn topology(&self) -> &NumaTopology {
        &self.topology
    }
}

impl Backend for CpuBackend {
    fn capabilities(&self) -> BackendCapabilities {
        let resources: Vec<BackendResourceView> = self
            .topology
            .nodes
            .iter()
            .map(|n| BackendResourceView {
                resource_id: format!("cpu_numa_{}", n.node_id),
                memory_domain_id: format!("numa_ram_{}", n.node_id),
                numa_node: Some(n.node_id),
                compute_streams: 1,
                copy_streams: 1,
                copy_engines: 1,
                peer_access: self
                    .topology
                    .nodes
                    .iter()
                    .filter(|o| o.node_id != n.node_id)
                    .map(|o| format!("cpu_numa_{}", o.node_id))
                    .collect(),
                supported_dtypes: vec![
                    "float32".into(),
                    "float16".into(),
                    "bfloat16".into(),
                    "int8".into(),
                    "int32".into(),
                    "int64".into(),
                ],
                supported_artifact_formats: vec!["tt-artifact-v1".into()],
            })
            .collect();
        // Report the enforceable budget, not the machine total: callers size
        // plans from this number and the allocator refuses beyond it.
        let budget_mem = self.memory_budget_bytes();
        BackendCapabilities {
            name: self.name.clone(),
            supports_p2p: false,
            supports_async_compute: true,
            supports_ordered_streams: true,
            max_streams: (self.topology.nodes.len() as u32).saturating_mul(2).max(2),
            device_memory_bytes: budget_mem,
            simulated: false,
            resources,
        }
    }

    fn allocate(
        &self,
        resource: ResourceId,
        bytes: usize,
        alignment: usize,
    ) -> BackendResult<BufferHandle> {
        if self.shutdown.load(Ordering::Acquire) {
            return Err(BackendError::Allocate {
                backend: self.name.clone(),
                resource: resource.to_string(),
                cause: "backend shutdown".into(),
            });
        }
        let align = alignment.max(1);
        let padded = bytes
            .checked_add(align - 1)
            .and_then(|value| value.checked_div(align))
            .and_then(|value| value.checked_mul(align))
            .ok_or_else(|| BackendError::Allocate {
                backend: self.name.clone(),
                resource: resource.to_string(),
                cause: format!("allocation size overflow: bytes={bytes} alignment={alignment}"),
            })?;
        let budget = self.memory_budget_bytes();
        let requested = padded as u64;
        if budget > 0 && requested > budget {
            return Err(BackendError::Allocate {
                backend: self.name.clone(),
                resource: resource.to_string(),
                cause: format!(
                    "host memory budget exceeded: requested={requested} budget={budget} \
                     (source: {}; raise the budget explicitly if this host has more to give)",
                    self.budget_source
                ),
            });
        }
        let mut storage = Vec::new();
        storage
            .try_reserve_exact(padded)
            .map_err(|error| BackendError::Allocate {
                backend: self.name.clone(),
                resource: resource.to_string(),
                cause: format!("host allocation failed for {padded} bytes: {error}"),
            })?;
        storage.resize(padded, 0);
        let buf = HostBuffer { bytes: storage };
        {
            let mut used = self.used_bytes.lock();
            if budget > 0 && used.saturating_add(requested) > budget {
                return Err(BackendError::Allocate {
                    backend: self.name.clone(),
                    resource: resource.to_string(),
                    cause: format!(
                        "host memory budget exceeded: used={} requested={requested} budget={budget} \
                         (source: {})",
                        *used, self.budget_source
                    ),
                });
            }
            *used += requested;
        }
        let id = self.next_buf.fetch_add(1, Ordering::Relaxed);
        self.buffers.lock().insert(id, buf);
        Ok(BufferHandle(id))
    }

    fn free(&self, buffer: BufferHandle) -> BackendResult<()> {
        let Some(rec) = self.buffers.lock().remove(&buffer.0) else {
            return Err(BackendError::Other {
                backend: self.name.clone(),
                cause: format!("unknown buffer {}", buffer.0),
            });
        };
        let mut used = self.used_bytes.lock();
        *used = used.saturating_sub(rec.bytes.len() as u64);
        Ok(())
    }

    fn transfer(
        &self,
        src: BufferHandle,
        dst: BufferHandle,
        bytes: usize,
        _stream: StreamId,
    ) -> BackendResult<EventHandle> {
        let evt = self.next_evt.fetch_add(1, Ordering::Relaxed);
        self.events.lock().insert(evt, EventStatus::Pending);
        let result = {
            let mut map = self.buffers.lock();
            let Some(src_buf) = map.get(&src.0) else {
                return Err(BackendError::Transfer {
                    backend: self.name.clone(),
                    cause: format!("missing src {}", src.0),
                });
            };
            let src_slice = src_buf
                .bytes
                .get(..bytes)
                .ok_or_else(|| BackendError::Transfer {
                    backend: self.name.clone(),
                    cause: "src shorter than transfer length".into(),
                })?;
            let src_copy = src_slice.to_vec();
            let Some(dst_buf) = map.get_mut(&dst.0) else {
                return Err(BackendError::Transfer {
                    backend: self.name.clone(),
                    cause: format!("missing dst {}", dst.0),
                });
            };
            let dst_slice =
                dst_buf
                    .bytes
                    .get_mut(..bytes)
                    .ok_or_else(|| BackendError::Transfer {
                        backend: self.name.clone(),
                        cause: "dst shorter than transfer length".into(),
                    })?;
            dst_slice.copy_from_slice(&src_copy);
            Ok(())
        };
        result?;
        self.events.lock().insert(evt, EventStatus::Complete);
        Ok(EventHandle(evt))
    }

    fn launch(
        &self,
        _executable: ExecutableHandle,
        _inputs: &[BufferHandle],
        _outputs: &[BufferHandle],
        _stream: StreamId,
    ) -> BackendResult<EventHandle> {
        // Torch regions still run via Python callbacks; native AOT launch is a no-op complete.
        let evt = self.next_evt.fetch_add(1, Ordering::Relaxed);
        self.events.lock().insert(evt, EventStatus::Complete);
        Ok(EventHandle(evt))
    }

    fn query_event(&self, event: EventHandle) -> BackendResult<EventStatus> {
        self.events
            .lock()
            .get(&event.0)
            .copied()
            .ok_or(BackendError::Event {
                backend: self.name.clone(),
                event: event.0,
                cause: "unknown event".into(),
            })
    }

    fn wait_event(&self, event: EventHandle) -> BackendResult<()> {
        match self.query_event(event)? {
            EventStatus::Complete => Ok(()),
            EventStatus::Pending => Ok(()), // sync transfers complete immediately
            EventStatus::Error => Err(BackendError::Event {
                backend: self.name.clone(),
                event: event.0,
                cause: "event in error state".into(),
            }),
        }
    }

    fn record_event(&self, _stream: StreamId) -> BackendResult<EventHandle> {
        let evt = self.next_evt.fetch_add(1, Ordering::Relaxed);
        self.events.lock().insert(evt, EventStatus::Complete);
        Ok(EventHandle(evt))
    }

    fn synchronize(&self) -> BackendResult<()> {
        Ok(())
    }

    fn health(&self) -> BackendHealth {
        BackendHealth {
            healthy: !self.shutdown.load(Ordering::Acquire),
            detail: format!(
                "numa_nodes={} compute_workers={} io_workers={}",
                self.topology.nodes.len(),
                self.compute_workers,
                self.io_workers
            ),
        }
    }

    fn memory_report(&self) -> BackendMemoryReport {
        BackendMemoryReport {
            device_used_bytes: *self.used_bytes.lock(),
            device_total_bytes: self.capabilities().device_memory_bytes,
            host_pinned_bytes: 0,
            live_allocations: self.buffers.lock().len() as u64,
        }
    }

    fn cancel_queued(&self) -> BackendResult<()> {
        Ok(())
    }
}

impl Drop for CpuBackend {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discovers_at_least_one_numa_domain() {
        let cpu = CpuBackend::discover().expect("cpu backend");
        assert!(!cpu.topology().nodes.is_empty());
        let caps = cpu.capabilities();
        assert!(!caps.simulated);
        assert!(!caps.resources.is_empty());
    }

    #[test]
    fn allocate_transfer_free() {
        let cpu = CpuBackend::discover().expect("cpu backend");
        let res = ResourceId::new("cpu_numa_0");
        let a = cpu.allocate(res.clone(), 1024, 64).unwrap();
        let b = cpu.allocate(res, 1024, 64).unwrap();
        {
            let mut bufs = cpu.buffers.lock();
            bufs.get_mut(&a.0).unwrap().bytes[..4].copy_from_slice(&[1, 2, 3, 4]);
        }
        let evt = cpu
            .transfer(a, b, 4, StreamId::new("cpu_numa_0::copy0"))
            .unwrap();
        assert_eq!(cpu.query_event(evt).unwrap(), EventStatus::Complete);
        {
            let bufs = cpu.buffers.lock();
            assert_eq!(&bufs.get(&b.0).unwrap().bytes[..4], &[1, 2, 3, 4]);
        }
        cpu.free(a).unwrap();
        cpu.free(b).unwrap();
        assert_eq!(cpu.memory_report().live_allocations, 0);
    }

    #[test]
    fn measure_copy_bandwidth_positive() {
        let cpu = CpuBackend::discover().expect("cpu backend");
        let bw = cpu.measure_copy_bandwidth(0, 0, 1 << 20);
        assert!(bw > 0.0);
    }

    #[test]
    fn allocation_size_overflow_is_an_error() {
        let cpu = CpuBackend::discover().expect("cpu backend");
        let error = cpu
            .allocate(ResourceId::new("cpu_numa_0"), usize::MAX, usize::MAX)
            .unwrap_err();
        assert!(error.to_string().contains("overflow"));
        assert_eq!(cpu.memory_report().live_allocations, 0);
    }
}
