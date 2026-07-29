//! Native residency session exposed to Python: Rust owns metadata; handles are opaque ids.

use crate::context_py::PyNativeExecutionContext;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use streamcompiler_core::{AllocationId, ResourceId, TensorId};
use streamcompiler_memory::{AllocationTable, ResidencyStore, TensorMetadata};
use streamcompiler_runtime::NativeExecutionContext;

static ALLOC_SEQ: AtomicU64 = AtomicU64::new(1);

#[pyclass(module = "streamcompiler._native", name = "NativeResidencySession")]
pub struct NativeResidencySession {
    store: Arc<ResidencyStore>,
    allocations: Arc<AllocationTable>,
    /// When set, this session is a view into a shared execution context.
    context: Option<Arc<NativeExecutionContext>>,
    put_count: AtomicU64,
    release_count: AtomicU64,
    require_count: AtomicU64,
}

#[pymethods]
impl NativeResidencySession {
    #[new]
    fn new() -> Self {
        let allocations = Arc::new(AllocationTable::new());
        let store = Arc::new(ResidencyStore::new(Arc::clone(&allocations)));
        Self {
            store,
            allocations,
            context: None,
            put_count: AtomicU64::new(0),
            release_count: AtomicU64::new(0),
            require_count: AtomicU64::new(0),
        }
    }

    /// Bind this session to an existing [`NativeExecutionContext`] (same residency store).
    #[staticmethod]
    fn from_execution_context(ctx: &PyNativeExecutionContext) -> Self {
        let inner = ctx.inner();
        Self {
            store: inner.residency(),
            allocations: inner.allocations(),
            context: Some(Arc::clone(inner)),
            put_count: AtomicU64::new(0),
            release_count: AtomicU64::new(0),
            require_count: AtomicU64::new(0),
        }
    }

    #[getter]
    fn execution_id(&self) -> Option<u64> {
        self.context.as_ref().map(|c| c.execution_id.as_u64())
    }

    /// Register a resident copy. `handle_id` is an opaque Python-side tensor id.
    #[pyo3(signature = (tensor_id, resource_id, handle_id, nbytes, authoritative=true, shape=None, strides=None, storage_offset=0, dtype="", storage_nbytes=0, storage_id=None))]
    #[allow(clippy::too_many_arguments)]
    fn put(
        &self,
        tensor_id: &str,
        resource_id: &str,
        handle_id: u64,
        nbytes: u64,
        authoritative: bool,
        shape: Option<Vec<i64>>,
        strides: Option<Vec<i64>>,
        storage_offset: i64,
        dtype: &str,
        storage_nbytes: u64,
        storage_id: Option<String>,
    ) -> PyResult<u64> {
        let alloc = if let Some(ref sid) = storage_id {
            AllocationId::new(format!("stor-{sid}"))
        } else {
            AllocationId::new(format!("pyh-{}", ALLOC_SEQ.fetch_add(1, Ordering::Relaxed)))
        };
        let meta = TensorMetadata {
            nbytes,
            shape: shape.unwrap_or_default(),
            strides: strides.unwrap_or_default(),
            storage_offset,
            dtype: dtype.to_owned(),
            storage_nbytes: if storage_nbytes > 0 {
                storage_nbytes
            } else {
                nbytes
            },
            storage_id,
            alias_group: None,
        };
        let copy = if authoritative {
            self.store
                .put_opaque(
                    TensorId::new(tensor_id),
                    ResourceId::new(resource_id),
                    alloc,
                    meta,
                    None,
                    Some(handle_id),
                )
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
        } else {
            // Ensure a primary copy exists, then replicate without invalidating siblings.
            let tid = TensorId::new(tensor_id);
            if self.store.logical_version(&tid) == 0 {
                self.store
                    .put_opaque(
                        tid.clone(),
                        ResourceId::new(resource_id),
                        alloc.clone(),
                        meta,
                        None,
                        Some(handle_id),
                    )
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            }
            self.store
                .replicate(&tid, ResourceId::new(resource_id), alloc, None)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
        };
        self.put_count.fetch_add(1, Ordering::Relaxed);
        Ok(copy.version)
    }

    /// Alias an existing valid copy onto another resource (shared allocation).
    fn alias(&self, tensor_id: &str, src_resource: &str, dst_resource: &str) -> PyResult<u64> {
        let copy = self
            .store
            .alias_same_allocation(
                &TensorId::new(tensor_id),
                &ResourceId::new(src_resource),
                ResourceId::new(dst_resource),
            )
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        self.put_count.fetch_add(1, Ordering::Relaxed);
        Ok(copy.version)
    }

    /// Require a valid copy; returns opaque handle id. Missing/stale → error.
    fn require(&self, tensor_id: &str, resource_id: &str) -> PyResult<u64> {
        self.require_count.fetch_add(1, Ordering::Relaxed);
        self.store
            .external_handle(&TensorId::new(tensor_id), &ResourceId::new(resource_id))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    fn has(&self, tensor_id: &str, resource_id: &str) -> bool {
        self.store
            .get(&TensorId::new(tensor_id), &ResourceId::new(resource_id))
            .is_ok()
    }

    fn acquire_lease(&self, tensor_id: &str, resource_id: &str) -> PyResult<()> {
        self.store
            .acquire_lease(&TensorId::new(tensor_id), &ResourceId::new(resource_id))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    fn release_lease(&self, tensor_id: &str, resource_id: &str) -> PyResult<()> {
        self.store
            .release_lease(&TensorId::new(tensor_id), &ResourceId::new(resource_id))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Strict release: fails if missing, already freed, or leased.
    fn release(&self, tensor_id: &str, resource_id: &str) -> PyResult<u64> {
        let freed = self
            .store
            .release_copy(&TensorId::new(tensor_id), &ResourceId::new(resource_id))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        self.release_count.fetch_add(1, Ordering::Relaxed);
        Ok(freed)
    }

    fn logical_version(&self, tensor_id: &str) -> u64 {
        self.store.logical_version(&TensorId::new(tensor_id))
    }

    fn peak_bytes(&self) -> u64 {
        self.allocations.peak_bytes()
    }

    fn live_bytes(&self) -> u64 {
        self.allocations.live_bytes()
    }

    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("put_count", self.put_count.load(Ordering::Relaxed))?;
        d.set_item("release_count", self.release_count.load(Ordering::Relaxed))?;
        d.set_item("require_count", self.require_count.load(Ordering::Relaxed))?;
        d.set_item("peak_bytes", self.peak_bytes())?;
        d.set_item("live_bytes", self.live_bytes())?;
        d.set_item("native_residency", true)?;
        if let Some(id) = self.execution_id() {
            d.set_item("execution_id", id)?;
            d.set_item("shared_execution_context", true)?;
        } else {
            d.set_item("shared_execution_context", false)?;
        }
        Ok(d)
    }
}

#[pyfunction]
pub fn new_native_residency() -> NativeResidencySession {
    NativeResidencySession::new()
}
