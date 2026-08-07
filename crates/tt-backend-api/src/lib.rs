//! Backend traits. Core scheduling never imports CUDA/ROCm.

mod traits;

pub use traits::{
    Backend, BackendCapabilities, BackendError, BackendHealth, BackendMemoryReport,
    BackendResourceView, BackendResult, BufferHandle, EventHandle, EventStatus, ExecutableHandle,
};
