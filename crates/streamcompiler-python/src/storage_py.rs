//! PyO3 bindings for native pack reader + chunk cache.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use std::path::PathBuf;
use std::sync::Mutex;
use streamcompiler_storage::{ChunkCache, PackManifest, PackReader, TensorEntry};

#[pyclass(module = "streamcompiler._native", name = "NativePackReader")]
pub struct NativePackReader {
    inner: Mutex<PackReader>,
}

#[pymethods]
impl NativePackReader {
    #[staticmethod]
    fn open(path: &str, manifest_json: &str) -> PyResult<Self> {
        let manifest = PackManifest::from_json(manifest_json)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let reader = PackReader::open(PathBuf::from(path), manifest)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self {
            inner: Mutex::new(reader),
        })
    }

    fn names(&self) -> PyResult<Vec<String>> {
        let g = self
            .inner
            .lock()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(g.manifest().tensors.iter().map(|t| t.name.clone()).collect())
    }

    fn entry(&self, py: Python<'_>, name: &str) -> PyResult<PyObject> {
        let g = self
            .inner
            .lock()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let entry = g
            .manifest()
            .tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| PyValueError::new_err(format!("unknown tensor {name}")))?;
        entry_to_dict(py, entry)
    }

    /// Positional read; releases the GIL.
    fn pread(&self, py: Python<'_>, name: &str) -> PyResult<PyObject> {
        let bytes = py.allow_threads(|| {
            let mut g = self
                .inner
                .lock()
                .map_err(|e| e.to_string())?;
            g.pread(name).map_err(|e| e.to_string())
        })
        .map_err(PyRuntimeError::new_err)?;
        Ok(PyBytes::new(py, &bytes).into())
    }
}

fn entry_to_dict(py: Python<'_>, entry: &TensorEntry) -> PyResult<PyObject> {
    let d = PyDict::new(py);
    d.set_item("name", &entry.name)?;
    d.set_item("offset", entry.offset)?;
    d.set_item("length", entry.length)?;
    d.set_item("dtype", &entry.dtype)?;
    d.set_item("shape", &entry.shape)?;
    d.set_item("checksum_crc32", entry.checksum_crc32)?;
    Ok(d.into())
}

#[pyclass(module = "streamcompiler._native", name = "NativeChunkCache")]
pub struct NativeChunkCache {
    inner: ChunkCache,
}

#[pymethods]
impl NativeChunkCache {
    #[new]
    fn new(capacity_bytes: u64) -> Self {
        Self {
            inner: ChunkCache::new(capacity_bytes),
        }
    }

    fn get<'py>(&self, py: Python<'py>, key: &str) -> Option<Bound<'py, PyBytes>> {
        self.inner.get(key).map(|b| PyBytes::new(py, &b))
    }

    fn insert(&self, key: &str, data: &[u8]) {
        let _ = self.inner.insert(key, data.to_vec());
    }

    fn release(&self, key: &str) {
        self.inner.release(key);
    }

    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let (hits, misses, live_bytes) = self.inner.stats();
        let d = PyDict::new(py);
        d.set_item("hits", hits)?;
        d.set_item("misses", misses)?;
        d.set_item("live_bytes", live_bytes)?;
        Ok(d)
    }
}
