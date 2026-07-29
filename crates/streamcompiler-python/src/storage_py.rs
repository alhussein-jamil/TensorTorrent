//! PyO3 bindings for native pack reader, chunk cache, and streaming store.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use streamcompiler_storage::{ChunkCache, PackManifest, PackReader, StreamingStore, TensorEntry};

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
        Ok(g.manifest()
            .tensors
            .iter()
            .map(|t| t.name.clone())
            .collect())
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
        let bytes = py
            .allow_threads(|| {
                let mut g = self.inner.lock().map_err(|e| e.to_string())?;
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

/// Prefetch + byte cache + shared inflight reads. Tensorize in Python.
#[pyclass(module = "streamcompiler._native", name = "NativeStreamingStore")]
pub struct NativeStreamingStore {
    inner: Arc<StreamingStore>,
}

#[pymethods]
impl NativeStreamingStore {
    #[staticmethod]
    fn open(path: &str, manifest_json: &str, capacity_bytes: u64) -> PyResult<Self> {
        let store = StreamingStore::open(path, manifest_json, capacity_bytes)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self {
            inner: Arc::new(store),
        })
    }

    /// Queue pack keys for background positional reads (GIL released).
    fn prefetch(&self, py: Python<'_>, keys: Vec<String>) {
        let store = Arc::clone(&self.inner);
        py.allow_threads(|| store.prefetch(&keys));
    }

    /// Block until key bytes are cached; returns owned bytes (GIL released during wait/IO).
    fn acquire_bytes(&self, py: Python<'_>, key: &str) -> PyResult<PyObject> {
        let key = key.to_owned();
        let store = Arc::clone(&self.inner);
        let data = py
            .allow_threads(|| store.acquire_bytes(&key).map_err(|e| e.to_string()))
            .map_err(PyRuntimeError::new_err)?;
        Ok(PyBytes::new(py, &data).into())
    }

    fn release(&self, key: &str) {
        self.inner.release(key);
    }

    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let s = self.inner.stats();
        let d = PyDict::new(py);
        d.set_item("cache_hits", s.cache_hits)?;
        d.set_item("cache_misses", s.cache_misses)?;
        d.set_item("live_bytes", s.live_bytes)?;
        d.set_item("prefetch_hits", s.prefetch_hits)?;
        d.set_item("waits_for_prefetch", s.waits_for_prefetch)?;
        d.set_item("bytes_read", s.bytes_read)?;
        d.set_item("prefetch_submitted", s.prefetch_submitted)?;
        d.set_item("native_streaming", s.native_streaming)?;
        Ok(d)
    }

    fn close(&self) {
        self.inner.close();
    }

    /// Timed native pread windows as ``(start_s, end_s, nbytes)`` relative to last reset.
    fn io_intervals(&self) -> Vec<(f64, f64, u64)> {
        self.inner.io_intervals()
    }

    fn reset_io_origin(&self) {
        self.inner.reset_io_origin();
    }
}

impl NativeStreamingStore {
    pub(crate) fn shared(&self) -> Arc<StreamingStore> {
        Arc::clone(&self.inner)
    }
}

#[pyfunction]
pub fn write_activation_spill(
    dir: &str,
    dtype: &str,
    shape: Vec<i64>,
    bytes: &[u8],
) -> PyResult<String> {
    let meta = streamcompiler_storage::SpillMeta {
        dtype: dtype.to_owned(),
        shape,
        nbytes: bytes.len() as u64,
    };
    let path = streamcompiler_storage::write_activation_spill(Path::new(dir), &meta, bytes)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(path.to_string_lossy().into_owned())
}

#[pyfunction]
pub fn read_activation_spill(py: Python<'_>, path: &str) -> PyResult<PyObject> {
    let (meta, bytes) = streamcompiler_storage::read_activation_spill(Path::new(path))
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let d = PyDict::new(py);
    d.set_item("dtype", meta.dtype)?;
    d.set_item("shape", meta.shape)?;
    d.set_item("nbytes", meta.nbytes)?;
    d.set_item("bytes", PyBytes::new(py, &bytes))?;
    Ok(d.into())
}

#[pyfunction]
pub fn remove_activation_spill(path: &str) -> PyResult<()> {
    streamcompiler_storage::remove_activation_spill(Path::new(path))
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))
}
