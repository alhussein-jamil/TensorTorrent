//! Python bindings for the production CPU backend.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::Arc;
use tt_backend_api::Backend;
use tt_backend_cpu::CpuBackend;
use tt_ir::ResourceId;

#[pyclass(module = "tensortorrent._native", name = "NativeCpuBackend")]
pub struct NativeCpuBackend {
    inner: Arc<CpuBackend>,
}

#[pymethods]
impl NativeCpuBackend {
    /// Discover the CPU backend sized from the effective host budget.
    ///
    /// Optional overrides come from the Python budget resolver; when absent,
    /// worker counts and the memory ceiling respect cgroup limits, scheduler
    /// affinity, and live availability instead of raw machine totals.
    #[staticmethod]
    #[pyo3(signature = (compute_workers=None, io_workers=None, memory_budget_bytes=None))]
    fn discover(
        compute_workers: Option<usize>,
        io_workers: Option<usize>,
        memory_budget_bytes: Option<u64>,
    ) -> PyResult<Self> {
        // Cap OpenMP/MKL/torch threads to the resolved compute worker count.
        let topo = tt_backend_cpu::discover_numa_topology();
        let budget = tt_backend_cpu::effective_host_budget();
        let effective_cores = topo
            .total_logical_cpus()
            .max(1)
            .min(budget.cpu_count.max(1));
        let intra = compute_workers
            .unwrap_or((effective_cores / 2).max(1))
            .max(1);
        CpuBackend::apply_thread_env_guards(intra, 1);
        let limits = tt_backend_cpu::CpuBackendLimits {
            compute_workers,
            io_workers,
            memory_budget_bytes,
        };
        let backend = CpuBackend::try_from_topology_with_limits(topo, limits)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self {
            inner: Arc::new(backend),
        })
    }

    /// Budget provenance for doctor/diagnostics.
    fn budget_report<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("memory_budget_bytes", self.inner.memory_budget_bytes())?;
        d.set_item("memory_budget_source", self.inner.memory_budget_source())?;
        Ok(d)
    }

    /// Override the enforced host memory ceiling (from the Python resolver).
    fn set_memory_budget_bytes(&self, bytes: u64) {
        self.inner.set_memory_budget_bytes(bytes);
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
            .free(tt_backend_api::BufferHandle(buffer))
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
