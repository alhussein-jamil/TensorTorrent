//! PyO3 wrapper around [`streamcompiler_runtime::NativeExecutionContext`].

use crate::artifact::NativeCancelToken;
use crate::storage_py::NativeStreamingStore;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use streamcompiler_backend_api::Backend;
use streamcompiler_runtime::NativeExecutionContext;

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
            .allocate(
                streamcompiler_core::ResourceId::new(resource),
                data.len().max(1),
                64,
            )
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
            .read_bytes(streamcompiler_backend_api::BufferHandle(buffer))
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(pyo3::types::PyBytes::new(py, &bytes))
    }

    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("execution_id", self.execution_id())?;
        d.set_item("peak_bytes", self.peak_bytes())?;
        d.set_item("live_bytes", self.live_bytes())?;
        d.set_item("cancelled", self.is_cancelled())?;
        d.set_item("has_streaming", self.inner.streaming_store().is_some())?;
        Ok(d)
    }
}
