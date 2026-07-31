//! Backend traits. Core scheduling never imports CUDA/ROCm.

mod abi;
mod traits;

pub use abi::{
    sc_backend_allocate, sc_backend_capabilities, sc_backend_launch, sc_backend_query_event,
    sc_backend_transfer, sc_backend_wait_event, ScBackendCapabilities, ScBufferHandle,
    ScEventHandle, ScEventStatus, ScResult, SC_ERR, SC_EVENT_COMPLETE, SC_EVENT_ERROR,
    SC_EVENT_PENDING, SC_OK,
};
pub use traits::{
    Backend, BackendCapabilities, BackendError, BackendHealth, BackendMemoryReport,
    BackendResourceView, BackendResult, BufferHandle, EventHandle, EventStatus, ExecutableHandle,
};
