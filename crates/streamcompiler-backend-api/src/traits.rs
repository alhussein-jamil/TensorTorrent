//! Opaque-handle backend trait for future CUDA/ROCm/vendor implementations.

use streamcompiler_core::{ResourceId, StreamId};
use thiserror::Error;

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
}
