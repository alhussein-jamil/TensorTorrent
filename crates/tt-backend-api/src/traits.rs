//! Opaque-handle backend trait. Core scheduling never imports CUDA/ROCm.

use thiserror::Error;
use tt_ir::{ResourceId, StreamId};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BufferHandle(pub u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EventHandle(pub u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ExecutableHandle(pub u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EventStatus {
    Pending,
    Complete,
    Error,
}

/// Memory / interconnect / compute surface exposed by a backend resource.
#[derive(Clone, Debug, Default)]
pub struct BackendResourceView {
    pub resource_id: String,
    pub memory_domain_id: String,
    pub numa_node: Option<u32>,
    pub compute_streams: u32,
    pub copy_streams: u32,
    pub copy_engines: u32,
    pub peer_access: Vec<String>,
    pub supported_dtypes: Vec<String>,
    pub supported_artifact_formats: Vec<String>,
}

#[derive(Clone, Debug, Default)]
pub struct BackendCapabilities {
    pub name: String,
    pub supports_p2p: bool,
    pub supports_async_compute: bool,
    pub supports_ordered_streams: bool,
    pub max_streams: u32,
    pub device_memory_bytes: u64,
    /// True when this backend is a deterministic simulator, not real silicon.
    pub simulated: bool,
    pub resources: Vec<BackendResourceView>,
}

#[derive(Clone, Debug, Default)]
pub struct BackendHealth {
    pub healthy: bool,
    pub detail: String,
}

#[derive(Clone, Debug, Default)]
pub struct BackendMemoryReport {
    pub device_used_bytes: u64,
    pub device_total_bytes: u64,
    pub host_pinned_bytes: u64,
    pub live_allocations: u64,
}

#[derive(Debug, Error)]
pub enum BackendError {
    #[error("backend {backend}: allocate failed on {resource}: {cause}")]
    Allocate {
        backend: String,
        resource: String,
        cause: String,
    },
    #[error("backend {backend}: transfer failed: {cause}")]
    Transfer { backend: String, cause: String },
    #[error("backend {backend}: launch failed: {cause}")]
    Launch { backend: String, cause: String },
    #[error("backend {backend}: event {event}: {cause}")]
    Event {
        backend: String,
        event: u64,
        cause: String,
    },
    #[error("backend {backend}: {cause}")]
    Other { backend: String, cause: String },
}

pub type BackendResult<T> = Result<T, BackendError>;

/// Device-agnostic backend contract used by the Rust runtime.
pub trait Backend: Send + Sync {
    fn capabilities(&self) -> BackendCapabilities;

    fn allocate(
        &self,
        resource: ResourceId,
        bytes: usize,
        alignment: usize,
    ) -> BackendResult<BufferHandle>;

    fn free(&self, buffer: BufferHandle) -> BackendResult<()>;

    fn transfer(
        &self,
        src: BufferHandle,
        dst: BufferHandle,
        bytes: usize,
        stream: StreamId,
    ) -> BackendResult<EventHandle>;

    fn launch(
        &self,
        executable: ExecutableHandle,
        inputs: &[BufferHandle],
        outputs: &[BufferHandle],
        stream: StreamId,
    ) -> BackendResult<EventHandle>;

    fn query_event(&self, event: EventHandle) -> BackendResult<EventStatus>;

    fn wait_event(&self, event: EventHandle) -> BackendResult<()>;

    /// Record a completion event on `stream` (optional backends may no-op).
    fn record_event(&self, stream: StreamId) -> BackendResult<EventHandle> {
        let _ = stream;
        Err(BackendError::Other {
            backend: self.capabilities().name,
            cause: "record_event unsupported".into(),
        })
    }

    fn synchronize(&self) -> BackendResult<()> {
        Ok(())
    }

    fn health(&self) -> BackendHealth {
        BackendHealth {
            healthy: true,
            detail: "ok".into(),
        }
    }

    fn memory_report(&self) -> BackendMemoryReport {
        let caps = self.capabilities();
        BackendMemoryReport {
            device_used_bytes: 0,
            device_total_bytes: caps.device_memory_bytes,
            host_pinned_bytes: 0,
            live_allocations: 0,
        }
    }

    fn cancel_queued(&self) -> BackendResult<()> {
        Ok(())
    }
}
