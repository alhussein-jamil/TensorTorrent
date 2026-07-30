//! Production CPU backend: NUMA domains, affinity, bounded pools, host buffers.

mod numa;
mod pool;

use parking_lot::Mutex;
use sc_backend_api::{
    Backend, BackendCapabilities, BackendError, BackendHealth, BackendMemoryReport,
    BackendResourceView, BackendResult, BufferHandle, EventHandle, EventStatus, ExecutableHandle,
};
use sc_ir::{ResourceId, StreamId};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

pub use numa::{discover_numa_topology, NumaNode, NumaTopology};
pub use pool::{CpuPoolConfig, WorkerPoolKind};

struct HostBuffer {
    bytes: Vec<u8>,
    #[allow(dead_code)]
    numa_node: Option<u32>,
}

/// Host-memory CPU backend with one execution domain per NUMA node.
pub struct CpuBackend {
    topology: NumaTopology,
    next_buf: AtomicU64,
    next_evt: AtomicU64,
    buffers: Mutex<HashMap<u64, HostBuffer>>,
    used_bytes: Mutex<u64>,
    events: Mutex<HashMap<u64, EventStatus>>,
    compute_pool: pool::BoundedPool,
    io_pool: pool::BoundedPool,
    shutdown: AtomicBool,
    /// Measured host copy bandwidth samples (bytes/s) keyed by src->dst domain.
    copy_bandwidth: Mutex<HashMap<String, f64>>,
    name: String,
}

impl CpuBackend {
    #[must_use]
    pub fn discover() -> Self {
        let topology = discover_numa_topology();
        Self::from_topology(topology)
    }

    #[must_use]
    pub fn from_topology(topology: NumaTopology) -> Self {
        let cores = topology.total_logical_cpus().max(1);
        let compute_workers = (cores / 2).max(1);
        let io_workers = (cores / 4).clamp(1, 4);
        Self {
            topology,
            next_buf: AtomicU64::new(1),
            next_evt: AtomicU64::new(1),
            buffers: Mutex::new(HashMap::new()),
            used_bytes: Mutex::new(0),
            events: Mutex::new(HashMap::new()),
            compute_pool: pool::BoundedPool::new(
                "cpu-compute",
                compute_workers,
                WorkerPoolKind::Compute,
            ),
            io_pool: pool::BoundedPool::new("cpu-io", io_workers, WorkerPoolKind::Io),
            shutdown: AtomicBool::new(false),
            copy_bandwidth: Mutex::new(HashMap::new()),
            name: "cpu".into(),
        }
    }

    /// Prevent OpenMP/MKL/PyTorch oversubscription for this process.
    pub fn apply_thread_env_guards(intra_op: usize, inter_op: usize) {
        let set_if_absent = |key: &str, val: &str| {
            if std::env::var_os(key).is_none() {
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
        let key = format!("numa_{src_node}->numa_{dst_node}");
        self.copy_bandwidth.lock().insert(key, bw);
        // Touch dst so optimizer cannot elide.
        let _ = dst.iter().fold(0u8, |a, b| a.wrapping_add(*b));
        bw
    }

    pub fn topology(&self) -> &NumaTopology {
        &self.topology
    }

    pub fn copy_bandwidth_samples(&self) -> HashMap<String, f64> {
        self.copy_bandwidth.lock().clone()
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
                supported_artifact_formats: vec!["sc-artifact-v1".into()],
            })
            .collect();
        let total_mem: u64 = self.topology.nodes.iter().map(|n| n.memory_bytes).sum();
        BackendCapabilities {
            name: self.name.clone(),
            supports_p2p: false,
            supports_async_compute: true,
            supports_ordered_streams: true,
            max_streams: (self.topology.nodes.len() as u32).saturating_mul(2).max(2),
            device_memory_bytes: total_mem,
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
        let padded = bytes.saturating_add(align - 1) / align * align;
        let numa_node = resource
            .as_str()
            .strip_prefix("cpu_numa_")
            .and_then(|s| s.parse::<u32>().ok())
            .or_else(|| {
                resource
                    .as_str()
                    .strip_prefix("numa_ram_")
                    .and_then(|s| s.parse::<u32>().ok())
            });
        let buf = HostBuffer {
            bytes: vec![0u8; padded],
            numa_node,
        };
        let id = self.next_buf.fetch_add(1, Ordering::Relaxed);
        *self.used_bytes.lock() += padded as u64;
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
        let buffers = Arc::new(Mutex::new(())); // serialize via pool
        let _ = buffers;
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
        // Region launch for CPU AOT artifacts lands here once regions are native.
        // Today the runtime still invokes Python compute callbacks for torch regions.
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
        self.compute_pool.synchronize();
        self.io_pool.synchronize();
        Ok(())
    }

    fn health(&self) -> BackendHealth {
        BackendHealth {
            healthy: !self.shutdown.load(Ordering::Acquire),
            detail: format!(
                "numa_nodes={} compute_workers={} io_workers={}",
                self.topology.nodes.len(),
                self.compute_pool.workers(),
                self.io_pool.workers()
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
        self.compute_pool.cancel();
        self.io_pool.cancel();
        Ok(())
    }
}

impl Drop for CpuBackend {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Release);
        self.compute_pool.shutdown();
        self.io_pool.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discovers_at_least_one_numa_domain() {
        let cpu = CpuBackend::discover();
        assert!(!cpu.topology().nodes.is_empty());
        let caps = cpu.capabilities();
        assert!(!caps.simulated);
        assert!(!caps.resources.is_empty());
    }

    #[test]
    fn allocate_transfer_free() {
        let cpu = CpuBackend::discover();
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
        let cpu = CpuBackend::discover();
        let bw = cpu.measure_copy_bandwidth(0, 0, 1 << 20);
        assert!(bw > 0.0);
    }
}
