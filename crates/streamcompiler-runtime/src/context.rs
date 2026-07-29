//! One authoritative execution context per forward.
//!
//! Scheduler, residency, events, allocations, storage spills, and cancellation
//! share this structure. Immutable artifact data is referenced, never mutated.

use parking_lot::Mutex;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use streamcompiler_memory::{AllocationTable, ResidencyStore};
use streamcompiler_storage::StreamingStore;
use streamcompiler_virtual_backend::{VirtualBackend, VirtualBackendConfig};

use crate::resources::ResourceState;

static EXECUTION_ID_SEQ: AtomicU64 = AtomicU64::new(1);

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
    /// tensor_id → spill file path
    pub spills: HashMap<String, PathBuf>,
    pub bytes_written: u64,
    pub bytes_read: u64,
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

    pub(crate) fn next_alloc_id(&self) -> streamcompiler_core::AllocationId {
        let n = self.alloc_counter.fetch_add(1, Ordering::Relaxed);
        streamcompiler_core::AllocationId::new(format!("ex-{}-{}", self.execution_id.as_u64(), n))
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

    /// Simulated accelerator for `resource` (created once per forward).
    pub fn virtual_backend(&self, resource: &str) -> Arc<VirtualBackend> {
        let mut map = self.virtual_backends.lock();
        if let Some(be) = map.get(resource) {
            return Arc::clone(be);
        }
        let be = Arc::new(VirtualBackend::new(VirtualBackendConfig {
            name: resource.to_owned(),
            ..Default::default()
        }));
        map.insert(resource.to_owned(), Arc::clone(&be));
        be
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
}
