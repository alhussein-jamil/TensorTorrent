//! PyO3 wrapper around [`streamcompiler_runtime::NativeExecutionContext`].

use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use streamcompiler_runtime::NativeExecutionContext;

use crate::artifact::NativeCancelToken;

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

    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("execution_id", self.execution_id())?;
        d.set_item("peak_bytes", self.peak_bytes())?;
        d.set_item("live_bytes", self.live_bytes())?;
        d.set_item("cancelled", self.is_cancelled())?;
        Ok(d)
    }
}
