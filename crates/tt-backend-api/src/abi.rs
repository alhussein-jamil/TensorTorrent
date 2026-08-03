//! C-compatible backend ABI for vendor implementations that cannot use the Rust trait.

use std::os::raw::{c_char, c_int, c_void};

pub const SC_OK: c_int = 0;
pub const SC_ERR: c_int = -1;
pub const SC_EVENT_PENDING: c_int = 0;
pub const SC_EVENT_COMPLETE: c_int = 1;
pub const SC_EVENT_ERROR: c_int = 2;

pub type TtResult = c_int;
pub type TtBufferHandle = u64;
pub type TtEventHandle = u64;
pub type TtEventStatus = c_int;

#[repr(C)]
pub struct TtBackendCapabilities {
    pub supports_p2p: c_int,
    pub supports_async_compute: c_int,
    pub supports_ordered_streams: c_int,
    pub max_streams: u32,
    pub device_memory_bytes: u64,
    pub simulated: c_int,
    pub name: *const c_char,
}

/// Opaque backend object owned by the vendor library.
pub type TtBackend = c_void;

/// Function pointer table a vendor can export. Not linked unless a vendor provides it.
#[repr(C)]
#[allow(dead_code)]
pub struct TtBackendVTable {
    pub capabilities: Option<
        unsafe extern "C" fn(backend: *mut TtBackend, out: *mut TtBackendCapabilities) -> TtResult,
    >,
    pub allocate: Option<
        unsafe extern "C" fn(
            backend: *mut TtBackend,
            resource: *const c_char,
            bytes: usize,
            alignment: usize,
            out: *mut TtBufferHandle,
        ) -> TtResult,
    >,
    pub transfer: Option<
        unsafe extern "C" fn(
            backend: *mut TtBackend,
            src: TtBufferHandle,
            dst: TtBufferHandle,
            bytes: usize,
            stream: *const c_char,
            out: *mut TtEventHandle,
        ) -> TtResult,
    >,
    pub launch: Option<
        unsafe extern "C" fn(
            backend: *mut TtBackend,
            executable: u64,
            inputs: *const TtBufferHandle,
            n_inputs: usize,
            outputs: *const TtBufferHandle,
            n_outputs: usize,
            stream: *const c_char,
            out: *mut TtEventHandle,
        ) -> TtResult,
    >,
    pub query_event: Option<
        unsafe extern "C" fn(
            backend: *mut TtBackend,
            event: TtEventHandle,
            out: *mut TtEventStatus,
        ) -> TtResult,
    >,
    pub wait_event:
        Option<unsafe extern "C" fn(backend: *mut TtBackend, event: TtEventHandle) -> TtResult>,
}

/// Stub exports so the ABI symbols exist for linking experiments.
///
/// # Safety
/// Caller must pass valid pointers. These stubs always return `SC_ERR`.
#[no_mangle]
pub unsafe extern "C" fn tt_backend_capabilities(
    _backend: *mut TtBackend,
    _out: *mut TtBackendCapabilities,
) -> TtResult {
    SC_ERR
}

/// # Safety
/// Caller must pass valid pointers.
#[no_mangle]
pub unsafe extern "C" fn tt_backend_allocate(
    _backend: *mut TtBackend,
    _resource: *const c_char,
    _bytes: usize,
    _alignment: usize,
    _out: *mut TtBufferHandle,
) -> TtResult {
    SC_ERR
}

/// # Safety
/// Caller must pass valid pointers.
#[no_mangle]
pub unsafe extern "C" fn tt_backend_transfer(
    _backend: *mut TtBackend,
    _src: TtBufferHandle,
    _dst: TtBufferHandle,
    _bytes: usize,
    _stream: *const c_char,
    _out: *mut TtEventHandle,
) -> TtResult {
    SC_ERR
}

/// # Safety
/// Caller must pass valid pointers.
#[no_mangle]
pub unsafe extern "C" fn tt_backend_launch(
    _backend: *mut TtBackend,
    _executable: u64,
    _inputs: *const TtBufferHandle,
    _n_inputs: usize,
    _outputs: *const TtBufferHandle,
    _n_outputs: usize,
    _stream: *const c_char,
    _out: *mut TtEventHandle,
) -> TtResult {
    SC_ERR
}

/// # Safety
/// Caller must pass valid pointers.
#[no_mangle]
pub unsafe extern "C" fn tt_backend_query_event(
    _backend: *mut TtBackend,
    _event: TtEventHandle,
    _out: *mut TtEventStatus,
) -> TtResult {
    SC_ERR
}

/// # Safety
/// Caller must pass valid pointers.
#[no_mangle]
pub unsafe extern "C" fn tt_backend_wait_event(
    _backend: *mut TtBackend,
    _event: TtEventHandle,
) -> TtResult {
    SC_ERR
}
