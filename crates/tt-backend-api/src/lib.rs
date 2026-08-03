//! Backend traits. Core scheduling never imports CUDA/ROCm.

mod abi;
mod traits;

pub use abi::{
    tt_backend_allocate, tt_backend_capabilities, tt_backend_launch, tt_backend_query_event,
    tt_backend_transfer, tt_backend_wait_event, TtBackendCapabilities, TtBufferHandle,
    TtEventHandle, TtEventStatus, TtResult, SC_ERR, SC_EVENT_COMPLETE, SC_EVENT_ERROR,
    SC_EVENT_PENDING, SC_OK,
};
pub use traits::{
    Backend, BackendCapabilities, BackendError, BackendHealth, BackendMemoryReport,
    BackendResourceView, BackendResult, BufferHandle, EventHandle, EventStatus, ExecutableHandle,
};
