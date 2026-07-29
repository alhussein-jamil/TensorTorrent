//! One authoritative execution context per forward.
//!
//! Scheduler, residency, events, allocations, and cancellation share this
//! structure. Immutable artifact data is referenced, never mutated.

use parking_lot::Mutex;
use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use streamcompiler_memory::{AllocationTable, ResidencyStore};

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

/// Authoritative mutable state for one schedule forward.
///
/// Created at the start of a forward; dropped when the forward completes.
/// Concurrent forwards use independent contexts sharing only immutable
/// artifact data.
#[derive(Debug)]
pub struct NativeExecutionContext {
    pub execution_id: ExecutionId,
    allocations: Arc<AllocationTable>,
    residency: Arc<ResidencyStore>,
    completed_events: Mutex<HashSet<String>>,
    alloc_counter: AtomicUsize,
    cancel: Arc<AtomicBool>,
    /// Explicit stream / copy-engine / link / I/O occupancy for this forward.
    resources: Mutex<ResourceState>,
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
        })
    }

    /// Build a context that reuses an existing residency store (PyO3 session).
    #[must_use]
    pub fn from_residency(
        residency: Arc<ResidencyStore>,
        cancel: Arc<AtomicBool>,
    ) -> Arc<Self> {
        let allocations = residency.allocations();
        Arc::new(Self {
            execution_id: ExecutionId(EXECUTION_ID_SEQ.fetch_add(1, Ordering::Relaxed)),
            allocations,
            residency,
            completed_events: Mutex::new(HashSet::new()),
            alloc_counter: AtomicUsize::new(0),
            cancel,
            resources: Mutex::new(ResourceState::new()),
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
        streamcompiler_core::AllocationId::new(format!(
            "ex-{}-{}",
            self.execution_id.as_u64(),
            n
        ))
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
}

impl Default for NativeExecutionContext {
    fn default() -> Self {
        // Prefer Arc::new via NativeExecutionContext::new(); Default for tests only.
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
