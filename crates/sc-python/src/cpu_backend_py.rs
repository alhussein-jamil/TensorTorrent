//! Python bindings for the production CPU backend.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use sc_backend_api::Backend;
use sc_backend_cpu::CpuBackend;
use sc_ir::ResourceId;
use std::sync::Arc;

#[pyclass(module = "streamcompiler._native", name = "NativeCpuBackend")]
pub struct NativeCpuBackend {
    inner: Arc<CpuBackend>,
}

#[pymethods]
impl NativeCpuBackend {
    #[staticmethod]
    fn discover() -> Self {
        // Bound pools; avoid oversubscription when torch/OpenMP also run.
        let topo = sc_backend_cpu::discover_numa_topology();
        let cores = topo.total_logical_cpus().max(1);
        CpuBackend::apply_thread_env_guards((cores / 2).max(1), 1);
        Self {
            inner: Arc::new(CpuBackend::from_topology(topo)),
        }
    }

    fn capabilities<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let caps = self.inner.capabilities();
        let d = PyDict::new(py);
        d.set_item("name", caps.name)?;
        d.set_item("simulated", caps.simulated)?;
        d.set_item("device_memory_bytes", caps.device_memory_bytes)?;
        d.set_item("max_streams", caps.max_streams)?;
        d.set_item("supports_async_compute", caps.supports_async_compute)?;
        let resources = pyo3::types::PyList::empty(py);
        for r in &caps.resources {
            let rd = PyDict::new(py);
            rd.set_item("resource_id", &r.resource_id)?;
            rd.set_item("memory_domain_id", &r.memory_domain_id)?;
            rd.set_item("numa_node", r.numa_node)?;
            rd.set_item("compute_streams", r.compute_streams)?;
            rd.set_item("copy_engines", r.copy_engines)?;
            resources.append(rd)?;
        }
        d.set_item("resources", resources)?;
        Ok(d)
    }

    fn health<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let h = self.inner.health();
        let d = PyDict::new(py);
        d.set_item("healthy", h.healthy)?;
        d.set_item("detail", h.detail)?;
        Ok(d)
    }

    fn memory_report<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let m = self.inner.memory_report();
        let d = PyDict::new(py);
        d.set_item("device_used_bytes", m.device_used_bytes)?;
        d.set_item("device_total_bytes", m.device_total_bytes)?;
        d.set_item("live_allocations", m.live_allocations)?;
        Ok(d)
    }

    fn allocate(&self, resource: &str, bytes: usize, alignment: usize) -> PyResult<u64> {
        self.inner
            .allocate(ResourceId::new(resource), bytes, alignment)
            .map(|h| h.0)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    fn free(&self, buffer: u64) -> PyResult<()> {
        self.inner
            .free(sc_backend_api::BufferHandle(buffer))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    fn measure_copy_bandwidth(&self, src_node: u32, dst_node: u32, nbytes: usize) -> f64 {
        self.inner
            .measure_copy_bandwidth(src_node, dst_node, nbytes)
    }

    fn numa_node_count(&self) -> usize {
        self.inner.topology().nodes.len()
    }
}
