//! Debug counters for native/Python boundary instrumentation.

use std::sync::atomic::AtomicU64;

pub(crate) static SCHEDULE_FROM_PY_CALLS: AtomicU64 = AtomicU64::new(0);
pub(crate) static SCHEDULER_ENTERS: AtomicU64 = AtomicU64::new(0);
pub(crate) static INSTRUCTION_CALLBACKS: AtomicU64 = AtomicU64::new(0);
pub(crate) static COMPUTE_CALLBACKS: AtomicU64 = AtomicU64::new(0);
pub(crate) static NON_COMPUTE_PYTHON_CALLBACKS: AtomicU64 = AtomicU64::new(0);
pub(crate) static GIL_ACQUISITIONS: AtomicU64 = AtomicU64::new(0);
pub(crate) static PARAMETER_LOAD_CALLBACKS: AtomicU64 = AtomicU64::new(0);
pub(crate) static PARAMETER_RELEASE_CALLBACKS: AtomicU64 = AtomicU64::new(0);
pub(crate) static SPILL_DEMATERIALIZE_CALLBACKS: AtomicU64 = AtomicU64::new(0);
pub(crate) static SPILL_MATERIALIZE_CALLBACKS: AtomicU64 = AtomicU64::new(0);
pub(crate) static HANDLE_RELEASE_CALLBACKS: AtomicU64 = AtomicU64::new(0);
pub(crate) static COPY_SYNC_CALLBACKS: AtomicU64 = AtomicU64::new(0);
pub(crate) static PYTHON_FALLBACK_ENTERS: AtomicU64 = AtomicU64::new(0);
pub(crate) static NATIVE_ARTIFACT_CREATED: AtomicU64 = AtomicU64::new(0);
