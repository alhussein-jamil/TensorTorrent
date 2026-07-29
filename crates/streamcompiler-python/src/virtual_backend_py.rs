//! PyO3 bindings for the async simulated virtual accelerator.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::Arc;
use streamcompiler_backend_api::{Backend, EventStatus};
use streamcompiler_core::{ResourceId, StreamId};
use streamcompiler_virtual_backend::{VirtualBackend, VirtualBackendConfig};

#[pyclass(module = "streamcompiler._native", name = "NativeVirtualBackend")]
pub struct NativeVirtualBackend {
    /// Arc — wait_event must not hold a process-wide mutex across sleep.
    inner: Arc<VirtualBackend>,
    name: String,
}

#[pymethods]
impl NativeVirtualBackend {
    #[new]
    #[pyo3(signature = (name="mock_accel0", memory_bytes=8u64<<30, compute_delay_s=0.05, transfer_bandwidth_bytes_per_s=12e9, transfer_latency_s=1e-5, max_copy_engines=2, supports_p2p=false))]
    fn new(
        name: &str,
        memory_bytes: u64,
        compute_delay_s: f64,
        transfer_bandwidth_bytes_per_s: f64,
        transfer_latency_s: f64,
        max_copy_engines: u32,
        supports_p2p: bool,
    ) -> Self {
        let config = VirtualBackendConfig {
            name: name.to_owned(),
            memory_bytes,
            compute_delay_s,
            transfer_bandwidth_bytes_per_s,
            transfer_latency_s,
            max_copy_engines,
            supports_p2p,
        };
        Self {
            inner: Arc::new(VirtualBackend::new(config)),
            name: name.to_owned(),
        }
    }

    /// All results from this backend are simulated — never claim hardware validation.
    fn capabilities<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let caps = self.inner.capabilities();
        let d = PyDict::new(py);
        d.set_item("name", &caps.name)?;
        d.set_item("supports_async_compute", caps.supports_async_compute)?;
        d.set_item("supports_ordered_streams", caps.supports_ordered_streams)?;
        d.set_item("max_streams", caps.max_streams)?;
        d.set_item("device_memory_bytes", caps.device_memory_bytes)?;
        d.set_item("simulated", caps.simulated)?;
        d.set_item("backend", &self.name)?;
        Ok(d)
    }

    fn allocate(&self, resource: &str, bytes: usize) -> PyResult<u64> {
        let h = self
            .inner
            .allocate(ResourceId::new(resource), bytes, 64)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(h.0)
    }

    fn free(&self, buffer: u64) -> PyResult<()> {
        self.inner
            .free(streamcompiler_backend_api::BufferHandle(buffer))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Submit async transfer; returns pending event id immediately (simulated).
    fn transfer(&self, src: u64, dst: u64, bytes: usize, stream: &str) -> PyResult<u64> {
        let ev = self
            .inner
            .transfer(
                streamcompiler_backend_api::BufferHandle(src),
                streamcompiler_backend_api::BufferHandle(dst),
                bytes,
                StreamId::new(stream),
            )
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(ev.0)
    }

    /// Submit async compute; returns pending event id immediately (simulated).
    fn launch(&self, stream: &str) -> PyResult<u64> {
        let ev = self
            .inner
            .launch(
                streamcompiler_backend_api::ExecutableHandle(0),
                &[],
                &[],
                StreamId::new(stream),
            )
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(ev.0)
    }

    fn query_event(&self, event: u64) -> PyResult<String> {
        let st = self
            .inner
            .query_event(streamcompiler_backend_api::EventHandle(event))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(match st {
            EventStatus::Pending => "pending".into(),
            EventStatus::Complete => "complete".into(),
            EventStatus::Error => "error".into(),
        })
    }

    /// Block until the simulated event completes (GIL released).
    fn wait_event(&self, event: u64) -> PyResult<()> {
        let inner = Arc::clone(&self.inner);
        Python::with_gil(|py| {
            py.allow_threads(|| {
                inner
                    .wait_event(streamcompiler_backend_api::EventHandle(event))
                    .map_err(|e| e.to_string())
            })
            .map_err(PyRuntimeError::new_err)
        })
    }

    fn shutdown(&self) {
        // Drop of last Arc joins workers; explicit no-op kept for API clarity.
    }

    /// Write host bytes into a native virtual buffer (distinct device storage).
    fn write_bytes(&self, buffer: u64, data: &[u8]) -> PyResult<()> {
        self.inner
            .write_bytes(streamcompiler_backend_api::BufferHandle(buffer), data)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Read native virtual buffer bytes.
    fn read_bytes<'py>(
        &self,
        py: Python<'py>,
        buffer: u64,
    ) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        let bytes = self
            .inner
            .read_bytes(streamcompiler_backend_api::BufferHandle(buffer))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(pyo3::types::PyBytes::new(py, &bytes))
    }
}

/// Smoke helper used by unit tests.
#[pyfunction]
pub fn virtual_backend_pending_is_async() -> PyResult<bool> {
    let be = VirtualBackend::new(VirtualBackendConfig {
        compute_delay_s: 0.05,
        ..Default::default()
    });
    let ev = be
        .launch(
            streamcompiler_backend_api::ExecutableHandle(0),
            &[],
            &[],
            StreamId::new("compute"),
        )
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let st = be
        .query_event(ev)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let pending = matches!(st, EventStatus::Pending);
    drop(be); // join workers
    Ok(pending)
}
