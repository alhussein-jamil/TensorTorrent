//! C-compatible backend ABI for vendor implementations that cannot use the Rust trait.

use std::os::raw::{c_char, c_int, c_void};

pub const SC_OK: c_int = 0;
pub const SC_ERR: c_int = -1;
pub const SC_EVENT_PENDING: c_int = 0;
pub const SC_EVENT_COMPLETE: c_int = 1;
pub const SC_EVENT_ERROR: c_int = 2;

pub type ScResult = c_int;
pub type ScBufferHandle = u64;
pub type ScEventHandle = u64;
pub type ScEventStatus = c_int;

#[repr(C)]
pub struct ScBackendCapabilities {
    pub supports_p2p: c_int,
    pub supports_async_compute: c_int,
    pub supports_ordered_streams: c_int,
    pub max_streams: u32,
    pub device_memory_bytes: u64,
    pub simulated: c_int,
    pub name: *const c_char,
}

/// Opaque backend object owned by the vendor library.
pub type ScBackend = c_void;

/// Function pointer table a vendor can export. Not linked unless a vendor provides it.
#[repr(C)]
#[allow(dead_code)]
pub struct ScBackendVTable {
    pub capabilities: Option<
        unsafe extern "C" fn(backend: *mut ScBackend, out: *mut ScBackendCapabilities) -> ScResult,
    >,
    pub allocate: Option<
        unsafe extern "C" fn(
            backend: *mut ScBackend,
            resource: *const c_char,
            bytes: usize,
            alignment: usize,
            out: *mut ScBufferHandle,
        ) -> ScResult,
    >,
    pub transfer: Option<
        unsafe extern "C" fn(
            backend: *mut ScBackend,
            src: ScBufferHandle,
            dst: ScBufferHandle,
            bytes: usize,
            stream: *const c_char,
            out: *mut ScEventHandle,
        ) -> ScResult,
    >,
    pub launch: Option<
        unsafe extern "C" fn(
            backend: *mut ScBackend,
            executable: u64,
            inputs: *const ScBufferHandle,
            n_inputs: usize,
            outputs: *const ScBufferHandle,
            n_outputs: usize,
            stream: *const c_char,
            out: *mut ScEventHandle,
        ) -> ScResult,
    >,
    pub query_event: Option<
        unsafe extern "C" fn(
            backend: *mut ScBackend,
            event: ScEventHandle,
            out: *mut ScEventStatus,
        ) -> ScResult,
    >,
    pub wait_event:
        Option<unsafe extern "C" fn(backend: *mut ScBackend, event: ScEventHandle) -> ScResult>,
}

/// Stub exports so the ABI symbols exist for linking experiments.
///
/// # Safety
/// Caller must pass valid pointers. These stubs always return `SC_ERR`.
#[no_mangle]
pub unsafe extern "C" fn sc_backend_capabilities(
    _backend: *mut ScBackend,
    _out: *mut ScBackendCapabilities,
) -> ScResult {
    SC_ERR
}

/// # Safety
/// Caller must pass valid pointers.
#[no_mangle]
pub unsafe extern "C" fn sc_backend_allocate(
    _backend: *mut ScBackend,
    _resource: *const c_char,
    _bytes: usize,
    _alignment: usize,
    _out: *mut ScBufferHandle,
) -> ScResult {
    SC_ERR
}

/// # Safety
/// Caller must pass valid pointers.
#[no_mangle]
pub unsafe extern "C" fn sc_backend_transfer(
    _backend: *mut ScBackend,
    _src: ScBufferHandle,
    _dst: ScBufferHandle,
    _bytes: usize,
    _stream: *const c_char,
    _out: *mut ScEventHandle,
) -> ScResult {
    SC_ERR
}

/// # Safety
/// Caller must pass valid pointers.
#[no_mangle]
pub unsafe extern "C" fn sc_backend_launch(
    _backend: *mut ScBackend,
    _executable: u64,
    _inputs: *const ScBufferHandle,
    _n_inputs: usize,
    _outputs: *const ScBufferHandle,
    _n_outputs: usize,
    _stream: *const c_char,
    _out: *mut ScEventHandle,
) -> ScResult {
    SC_ERR
}

/// # Safety
/// Caller must pass valid pointers.
#[no_mangle]
pub unsafe extern "C" fn sc_backend_query_event(
    _backend: *mut ScBackend,
    _event: ScEventHandle,
    _out: *mut ScEventStatus,
) -> ScResult {
    SC_ERR
}

/// # Safety
/// Caller must pass valid pointers.
#[no_mangle]
pub unsafe extern "C" fn sc_backend_wait_event(
    _backend: *mut ScBackend,
    _event: ScEventHandle,
) -> ScResult {
    SC_ERR
}
