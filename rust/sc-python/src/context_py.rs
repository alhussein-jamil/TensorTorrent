//! PyO3 wrapper around [`sc_runtime::NativeExecutionContext`].

use crate::artifact::NativeCancelToken;
use crate::storage_py::NativeStreamingStore;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use sc_backend_api::Backend;
use sc_runtime::NativeExecutionContext;
use std::collections::HashMap;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

/// One authoritative native execution context per forward.
#[pyclass(module = "streamcompiler._native", name = "NativeExecutionContext")]
pub struct PyNativeExecutionContext {
    inner: Arc<NativeExecutionContext>,
}

impl PyNativeExecutionContext {
    pub(crate) fn inner(&self) -> &Arc<NativeExecutionContext> {
        &self.inner
    }
}

#[pymethods]
impl PyNativeExecutionContext {
    #[new]
    #[pyo3(signature = (cancel_token=None))]
    fn new(cancel_token: Option<&NativeCancelToken>) -> Self {
        let cancel = cancel_token
            .map(NativeCancelToken::arc)
            .unwrap_or_else(|| Arc::new(AtomicBool::new(false)));
        Self {
            inner: NativeExecutionContext::with_cancel(cancel),
        }
    }

    #[getter]
    fn execution_id(&self) -> u64 {
        self.inner.execution_id.as_u64()
    }

    fn peak_bytes(&self) -> u64 {
        self.inner.peak_bytes()
    }

    fn live_bytes(&self) -> u64 {
        self.inner.live_bytes()
    }

    fn is_cancelled(&self) -> bool {
        self.inner.is_cancelled()
    }

    fn request_cancel(&self) {
        self.inner.request_cancel();
    }

    fn set_spill_dir(&self, dir: &str) {
        self.inner.set_spill_dir(std::path::PathBuf::from(dir));
    }

    /// Attach the process-wide pack streaming store for native Prefetch/Load.
    fn set_streaming_store(&self, store: &NativeStreamingStore, bindings: HashMap<String, String>) {
        self.inner.set_streaming(store.shared(), bindings);
    }

    /// Allocate + write a native virtual-device buffer (not a host alias).
    fn virtual_buffer_from_bytes(&self, resource: &str, data: &[u8]) -> PyResult<u64> {
        let be = self.inner.virtual_backend(resource);
        let h = be
            .allocate(sc_ir::ResourceId::new(resource), data.len().max(1), 64)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        be.write_bytes(h, data)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(h.0)
    }

    /// Read bytes from a native virtual-device buffer owned by this context.
    fn virtual_buffer_to_bytes<'py>(
        &self,
        py: Python<'py>,
        resource: &str,
        buffer: u64,
    ) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        let be = self.inner.virtual_backend(resource);
        let bytes = be
            .read_bytes(sc_backend_api::BufferHandle(buffer))
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(pyo3::types::PyBytes::new(py, &bytes))
    }

    /// Bind a virtual buffer to the allocation for `(tensor, resource)`.
    ///
    /// On final native allocation release the buffer is freed immediately.
    fn bind_virtual_buffer(&self, tensor_id: &str, resource: &str, buffer_id: u64) -> PyResult<()> {
        self.inner
            .bind_virtual_buffer(tensor_id, resource, buffer_id)
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)
    }

    /// Configure virtual-backend capacity/timing from topology (before first use).
    #[pyo3(signature = (resource, memory_bytes=None, transfer_bandwidth_bytes_per_s=None, transfer_latency_s=None, compute_delay_s=None))]
    fn set_virtual_backend_config(
        &self,
        resource: &str,
        memory_bytes: Option<u64>,
        transfer_bandwidth_bytes_per_s: Option<f64>,
        transfer_latency_s: Option<f64>,
        compute_delay_s: Option<f64>,
    ) {
        let mut cfg = sc_backend_virtual::VirtualBackendConfig {
            name: resource.to_owned(),
            ..Default::default()
        };
        if let Some(m) = memory_bytes {
            cfg.memory_bytes = m;
        }
        if let Some(bw) = transfer_bandwidth_bytes_per_s {
            cfg.transfer_bandwidth_bytes_per_s = bw;
        }
        if let Some(lat) = transfer_latency_s {
            cfg.transfer_latency_s = lat;
        }
        if let Some(d) = compute_delay_s {
            cfg.compute_delay_s = d;
        }
        self.inner.set_virtual_backend_config(resource, cfg);
    }

    fn virtual_backend_used_bytes(&self, resource: &str) -> u64 {
        self.inner.virtual_backend_used_bytes(resource)
    }

    fn virtual_backend_live_buffers(&self, resource: &str) -> usize {
        self.inner.virtual_backend_live_buffers(resource)
    }

    fn virtual_peak_bytes(&self) -> u64 {
        self.inner.virtual_peak_bytes()
    }

    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("execution_id", self.execution_id())?;
        d.set_item("peak_bytes", self.peak_bytes())?;
        d.set_item("live_bytes", self.live_bytes())?;
        d.set_item("virtual_peak_bytes", self.virtual_peak_bytes())?;
        d.set_item("cancelled", self.is_cancelled())?;
        d.set_item("has_streaming", self.inner.streaming_store().is_some())?;
        Ok(d)
    }
}
