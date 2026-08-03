//! One authoritative execution context per forward.
//!
//! Scheduler, residency, events, allocations, storage spills, and cancellation
//! share this structure. Immutable artifact data is referenced, never mutated.

use parking_lot::Mutex;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use tt_backend_api::Backend;
use tt_backend_virtual::{VirtualBackend, VirtualBackendConfig};
use tt_memory::{AllocationTable, ResidencyStore};
use tt_storage::StreamingStore;

use crate::resources::ResourceState;

static EXECUTION_ID_SEQ: AtomicU64 = AtomicU64::new(1);

/// Default progress-stall watchdog: 5 minutes of zero progress is treated as
/// a lost completion / deadlock rather than legitimately slow I/O.
pub const DEFAULT_STALL_TIMEOUT_MS: u64 = 300_000;

/// Opaque typed execution id (monotonic within process).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct ExecutionId(u64);

impl ExecutionId {
    #[must_use]
    pub fn as_u64(self) -> u64 {
        self.0
    }
}

/// Spill file bookkeeping owned by the execution context.
#[derive(Debug, Default)]
pub struct ExecutionStorageState {
    pub spill_dir: Option<PathBuf>,
    /// Per-execution session directory (`tt-spill-<pid>-<exec>`) once created.
    pub session_dir: Option<PathBuf>,
    /// tensor_id → spill file path
    pub spills: HashMap<String, PathBuf>,
    pub bytes_written: u64,
    pub bytes_read: u64,
    /// Aggregate cap across live spill bytes for this execution (None = uncapped).
    pub spill_budget_bytes: Option<u64>,
    /// Live (written, not yet reloaded) spill bytes counted against the budget.
    pub spill_live_bytes: u64,
}

/// Authoritative mutable state for one schedule forward.
///
/// Created at the start of a forward; dropped when the forward completes.
/// Concurrent forwards use independent contexts sharing only immutable
/// artifact data.
pub struct NativeExecutionContext {
    pub execution_id: ExecutionId,
    allocations: Arc<AllocationTable>,
    residency: Arc<ResidencyStore>,
    completed_events: Mutex<HashSet<String>>,
    alloc_counter: AtomicUsize,
    cancel: Arc<AtomicBool>,
    /// Explicit stream / copy-engine / link / I/O occupancy for this forward.
    resources: Mutex<ResourceState>,
    /// Activation spill files and related I/O counters.
    storage: Mutex<ExecutionStorageState>,
    /// Optional pack streaming store (shared with Python StreamingParameterStore).
    streaming: Mutex<Option<Arc<StreamingStore>>>,
    /// Environment tensor id → pack logical key.
    pack_bindings: Mutex<HashMap<String, String>>,
    /// Per-resource simulated accelerators (public mock path). Labelled simulated.
    virtual_backends: Mutex<HashMap<String, Arc<VirtualBackend>>>,
    /// Optional per-resource VirtualBackendConfig (memory/bandwidth from topology).
    virtual_backend_configs: Mutex<HashMap<String, VirtualBackendConfig>>,
    /// allocation_id → (resource, virtual buffer id) for prompt free on final ref.
    virtual_buffers: Mutex<HashMap<String, (String, u64)>>,
    /// Peak observed virtual-device used bytes across resources (leak diagnostics).
    virtual_peak_bytes: AtomicU64,
    /// Progress generation: bumped on every completion / resource release so
    /// waiters can distinguish "slow" from "stuck".
    progress_gen: AtomicU64,
    /// Stall watchdog in milliseconds (0 = disabled). Default 300 000.
    stall_timeout_ms: AtomicU64,
}

impl std::fmt::Debug for NativeExecutionContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NativeExecutionContext")
            .field("execution_id", &self.execution_id)
            .field("has_streaming", &self.streaming.lock().is_some())
            .field("pack_bindings", &self.pack_bindings.lock().len())
            .finish_non_exhaustive()
    }
}

impl NativeExecutionContext {
    #[must_use]
    pub fn new() -> Arc<Self> {
        Self::with_cancel(Arc::new(AtomicBool::new(false)))
    }

    #[must_use]
    pub fn with_cancel(cancel: Arc<AtomicBool>) -> Arc<Self> {
        let allocations = Arc::new(AllocationTable::new());
        let residency = Arc::new(ResidencyStore::new(Arc::clone(&allocations)));
        Arc::new(Self {
            execution_id: ExecutionId(EXECUTION_ID_SEQ.fetch_add(1, Ordering::Relaxed)),
            allocations,
            residency,
            completed_events: Mutex::new(HashSet::new()),
            alloc_counter: AtomicUsize::new(0),
            cancel,
            resources: Mutex::new(ResourceState::new()),
            storage: Mutex::new(ExecutionStorageState::default()),
            streaming: Mutex::new(None),
            pack_bindings: Mutex::new(HashMap::new()),
            virtual_backends: Mutex::new(HashMap::new()),
            virtual_backend_configs: Mutex::new(HashMap::new()),
            virtual_buffers: Mutex::new(HashMap::new()),
            virtual_peak_bytes: AtomicU64::new(0),
            progress_gen: AtomicU64::new(0),
            stall_timeout_ms: AtomicU64::new(DEFAULT_STALL_TIMEOUT_MS),
        })
    }

    /// Build a context that reuses an existing residency store (PyO3 session).
    #[must_use]
    pub fn from_residency(residency: Arc<ResidencyStore>, cancel: Arc<AtomicBool>) -> Arc<Self> {
        let allocations = residency.allocations();
        Arc::new(Self {
            execution_id: ExecutionId(EXECUTION_ID_SEQ.fetch_add(1, Ordering::Relaxed)),
            allocations,
            residency,
            completed_events: Mutex::new(HashSet::new()),
            alloc_counter: AtomicUsize::new(0),
            cancel,
            resources: Mutex::new(ResourceState::new()),
            storage: Mutex::new(ExecutionStorageState::default()),
            streaming: Mutex::new(None),
            pack_bindings: Mutex::new(HashMap::new()),
            virtual_backends: Mutex::new(HashMap::new()),
            virtual_backend_configs: Mutex::new(HashMap::new()),
            virtual_buffers: Mutex::new(HashMap::new()),
            virtual_peak_bytes: AtomicU64::new(0),
            progress_gen: AtomicU64::new(0),
            stall_timeout_ms: AtomicU64::new(DEFAULT_STALL_TIMEOUT_MS),
        })
    }

    #[must_use]
    pub fn allocations(&self) -> Arc<AllocationTable> {
        Arc::clone(&self.allocations)
    }

    #[must_use]
    pub fn residency(&self) -> Arc<ResidencyStore> {
        Arc::clone(&self.residency)
    }

    #[must_use]
    pub fn cancel_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.cancel)
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancel.load(Ordering::Acquire)
    }

    pub fn request_cancel(&self) {
        self.cancel.store(true, Ordering::Release);
    }

    pub(crate) fn completed_events(&self) -> &Mutex<HashSet<String>> {
        &self.completed_events
    }

    pub(crate) fn next_alloc_id(&self) -> tt_ir::AllocationId {
        let n = self.alloc_counter.fetch_add(1, Ordering::Relaxed);
        tt_ir::AllocationId::new(format!("ex-{}-{}", self.execution_id.as_u64(), n))
    }

    pub fn peak_bytes(&self) -> u64 {
        self.allocations.peak_bytes()
    }

    pub fn live_bytes(&self) -> u64 {
        self.allocations.live_bytes()
    }

    pub fn with_resources<R>(&self, f: impl FnOnce(&mut ResourceState) -> R) -> R {
        f(&mut self.resources.lock())
    }

    pub fn with_storage<R>(&self, f: impl FnOnce(&mut ExecutionStorageState) -> R) -> R {
        f(&mut self.storage.lock())
    }

    pub fn set_spill_dir(&self, dir: PathBuf) {
        self.storage.lock().spill_dir = Some(dir);
    }

    /// Cap aggregate live spill bytes for this execution.
    pub fn set_spill_budget_bytes(&self, bytes: u64) {
        self.storage.lock().spill_budget_bytes = Some(bytes);
    }

    /// Configure the progress-stall watchdog (0 disables it).
    pub fn set_stall_timeout_secs(&self, secs: f64) {
        let ms = if secs <= 0.0 {
            0
        } else {
            (secs * 1000.0).round().min(u64::MAX as f64) as u64
        };
        self.stall_timeout_ms.store(ms, Ordering::Release);
    }

    #[must_use]
    pub fn stall_timeout(&self) -> Option<std::time::Duration> {
        let ms = self.stall_timeout_ms.load(Ordering::Acquire);
        if ms == 0 {
            None
        } else {
            Some(std::time::Duration::from_millis(ms))
        }
    }

    /// Record forward progress (completion arrived, resource released) so
    /// stalled waiters reset their watchdog.
    pub fn bump_progress(&self) {
        self.progress_gen.fetch_add(1, Ordering::AcqRel);
    }

    #[must_use]
    pub fn progress_generation(&self) -> u64 {
        self.progress_gen.load(Ordering::Acquire)
    }

    /// Enforce a hard byte ceiling for a resource on the REAL allocation path
    /// (previously only the virtual/mock path ever set limits).
    pub fn set_resource_capacity(&self, resource: &str, bytes: u64) {
        self.allocations.set_capacity_limit(resource, bytes);
    }

    /// Lazily create this execution's spill session directory under the
    /// configured spill dir (or the system temp dir as a last resort).
    /// Refuses RAM-backed filesystems unless TT_ALLOW_TMPFS_SPILL=1.
    pub fn spill_session_dir(&self) -> Result<PathBuf, String> {
        if let Some(dir) = self.storage.lock().session_dir.clone() {
            return Ok(dir);
        }
        let base = self
            .storage
            .lock()
            .spill_dir
            .clone()
            .unwrap_or_else(std::env::temp_dir);
        let allow_ram = std::env::var("TT_ALLOW_TMPFS_SPILL").as_deref() == Ok("1");
        tt_storage::ensure_spill_dir_usable(&base, allow_ram).map_err(|e| e.to_string())?;
        let session = tt_storage::create_spill_session_dir(&base, self.execution_id.as_u64())
            .map_err(|e| e.to_string())?;
        self.storage.lock().session_dir = Some(session.clone());
        Ok(session)
    }

    pub fn set_streaming(&self, store: Arc<StreamingStore>, bindings: HashMap<String, String>) {
        store.reset_io_origin();
        *self.streaming.lock() = Some(store);
        *self.pack_bindings.lock() = bindings;
    }

    pub fn streaming_store(&self) -> Option<Arc<StreamingStore>> {
        self.streaming.lock().clone()
    }

    pub fn pack_key(&self, tensor_id: &str) -> String {
        self.pack_bindings
            .lock()
            .get(tensor_id)
            .cloned()
            .unwrap_or_else(|| tensor_id.to_owned())
    }

    /// Install topology-derived virtual-backend priors before first use.
    pub fn set_virtual_backend_config(&self, resource: &str, config: VirtualBackendConfig) {
        self.virtual_backend_configs
            .lock()
            .insert(resource.to_owned(), config);
    }

    /// Simulated accelerator for `resource` (created once per forward).
    pub fn virtual_backend(&self, resource: &str) -> Arc<VirtualBackend> {
        let mut map = self.virtual_backends.lock();
        if let Some(be) = map.get(resource) {
            return Arc::clone(be);
        }
        let config = self
            .virtual_backend_configs
            .lock()
            .get(resource)
            .cloned()
            .unwrap_or_else(|| VirtualBackendConfig {
                name: resource.to_owned(),
                ..Default::default()
            });
        let mut cfg = config;
        if cfg.name.is_empty() {
            cfg.name = resource.to_owned();
        }
        let be = Arc::new(VirtualBackend::new(cfg));
        map.insert(resource.to_owned(), Arc::clone(&be));
        be
    }

    /// Bind a virtual-device buffer to the allocation backing `(tensor, resource)`.
    pub fn bind_virtual_buffer(
        &self,
        tensor: &str,
        resource: &str,
        buffer_id: u64,
    ) -> Result<(), String> {
        use tt_ir::{ResourceId, TensorId};
        let copy = self
            .residency
            .get(&TensorId::new(tensor), &ResourceId::new(resource))
            .map_err(|e| e.to_string())?;
        let alloc_key = copy.allocation.as_str().to_owned();
        if let Some((prev_res, prev_buf)) = self
            .virtual_buffers
            .lock()
            .insert(alloc_key, (resource.to_owned(), buffer_id))
        {
            if prev_buf != buffer_id {
                if let Some(be) = self.virtual_backends.lock().get(&prev_res) {
                    let _ = be.free(tt_backend_api::BufferHandle(prev_buf));
                }
            }
        }
        // Capacity ceiling for mock device memory when configured.
        if let Some(limit) = self
            .virtual_backend_configs
            .lock()
            .get(resource)
            .map(|c| c.memory_bytes)
        {
            self.allocations.set_capacity_limit(resource, limit);
        }
        let used = self.virtual_backend_used_bytes(resource);
        self.virtual_peak_bytes.fetch_max(used, Ordering::Relaxed);
        Ok(())
    }

    #[must_use]
    pub fn virtual_peak_bytes(&self) -> u64 {
        self.virtual_peak_bytes.load(Ordering::Relaxed)
    }

    /// Free virtual buffer when the final allocation reference was dropped.
    pub fn free_virtual_buffer_for_alloc(&self, allocation_id: &str, freed_bytes: u64) {
        if freed_bytes == 0 {
            return;
        }
        let Some((resource, buffer_id)) = self.virtual_buffers.lock().remove(allocation_id) else {
            return;
        };
        let be = self.virtual_backend(&resource);
        let _ = be.free(tt_backend_api::BufferHandle(buffer_id));
    }

    /// Live virtual-device bytes for `resource` (0 if backend not yet created).
    pub fn virtual_backend_used_bytes(&self, resource: &str) -> u64 {
        let map = self.virtual_backends.lock();
        map.get(resource).map(|be| be.used_bytes()).unwrap_or(0)
    }

    /// Live virtual buffer count for `resource`.
    pub fn virtual_backend_live_buffers(&self, resource: &str) -> usize {
        let map = self.virtual_backends.lock();
        map.get(resource)
            .map(|be| be.live_buffer_count())
            .unwrap_or(0)
    }
}

impl Default for NativeExecutionContext {
    fn default() -> Self {
        let allocations = Arc::new(AllocationTable::new());
        let residency = Arc::new(ResidencyStore::new(Arc::clone(&allocations)));
        Self {
            execution_id: ExecutionId(EXECUTION_ID_SEQ.fetch_add(1, Ordering::Relaxed)),
            allocations,
            residency,
            completed_events: Mutex::new(HashSet::new()),
            alloc_counter: AtomicUsize::new(0),
            cancel: Arc::new(AtomicBool::new(false)),
            resources: Mutex::new(ResourceState::new()),
            storage: Mutex::new(ExecutionStorageState::default()),
            streaming: Mutex::new(None),
            pack_bindings: Mutex::new(HashMap::new()),
            virtual_backends: Mutex::new(HashMap::new()),
            virtual_backend_configs: Mutex::new(HashMap::new()),
            virtual_buffers: Mutex::new(HashMap::new()),
            virtual_peak_bytes: AtomicU64::new(0),
            progress_gen: AtomicU64::new(0),
            stall_timeout_ms: AtomicU64::new(DEFAULT_STALL_TIMEOUT_MS),
        }
    }
}

impl Drop for NativeExecutionContext {
    fn drop(&mut self) {
        // Spill files must not outlive the execution: crash-path cleanup is
        // handled by the startup orphan sweep, but every normal/cancelled/
        // errored forward cleans its own session here.
        {
            let mut st = self.storage.lock();
            st.spills.clear();
            if let Some(session) = st.session_dir.take() {
                tt_storage::remove_spill_session_dir(&session);
            }
        }
        // Schedules without explicit Release still must not leak device buffers.
        let pending: Vec<(String, u64)> = self
            .virtual_buffers
            .lock()
            .drain()
            .map(|(_, (resource, buf))| (resource, buf))
            .collect();
        for (resource, buf_id) in pending {
            if let Some(be) = self.virtual_backends.lock().get(&resource) {
                let _ = be.free(tt_backend_api::BufferHandle(buf_id));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn independent_contexts_have_distinct_ids() {
        let a = NativeExecutionContext::new();
        let b = NativeExecutionContext::new();
        assert_ne!(a.execution_id, b.execution_id);
        assert_eq!(a.live_bytes(), 0);
        assert_eq!(b.live_bytes(), 0);
    }

    #[test]
    fn from_residency_shares_store() {
        let allocations = Arc::new(AllocationTable::new());
        let store = Arc::new(ResidencyStore::new(Arc::clone(&allocations)));
        let ctx = NativeExecutionContext::from_residency(
            Arc::clone(&store),
            Arc::new(AtomicBool::new(false)),
        );
        assert!(Arc::ptr_eq(&ctx.residency(), &store));
    }

    #[test]
    fn final_alloc_release_frees_virtual_buffer() {
        use tt_backend_api::Backend;
        use tt_ir::{AllocationId, ResourceId, TensorId};
        use tt_memory::TensorMetadata;

        let ctx = NativeExecutionContext::new();
        ctx.set_virtual_backend_config(
            "mock_accel0",
            VirtualBackendConfig {
                name: "mock_accel0".into(),
                memory_bytes: 64 * 1024,
                ..Default::default()
            },
        );
        let be = ctx.virtual_backend("mock_accel0");
        let buf = be
            .allocate(ResourceId::new("mock_accel0"), 1024, 64)
            .unwrap();
        assert_eq!(be.used_bytes(), 1024);
        let tid = TensorId::new("t0");
        let rid = ResourceId::new("mock_accel0");
        let aid = AllocationId::new("a0");
        ctx.residency()
            .put(
                tid.clone(),
                rid.clone(),
                aid.clone(),
                TensorMetadata {
                    nbytes: 1024,
                    ..Default::default()
                },
                None,
            )
            .unwrap();
        ctx.bind_virtual_buffer("t0", "mock_accel0", buf.0).unwrap();
        let freed = ctx.residency().release_copy(&tid, &rid).unwrap();
        assert_eq!(freed, 1024);
        ctx.free_virtual_buffer_for_alloc(aid.as_str(), freed);
        assert_eq!(be.used_bytes(), 0);
        assert_eq!(be.live_buffer_count(), 0);
    }
}
