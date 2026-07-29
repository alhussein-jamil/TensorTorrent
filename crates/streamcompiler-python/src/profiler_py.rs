//! PyO3 bindings for native profile database.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::Mutex;
use streamcompiler_profiler::{CostStatus, ProfileDatabase, ProfileRecord, RegionCost};

#[pyclass(module = "streamcompiler._native", name = "NativeProfileDatabase")]
pub struct NativeProfileDatabase {
    inner: Mutex<ProfileDatabase>,
}

#[pymethods]
impl NativeProfileDatabase {
    #[new]
    fn new() -> Self {
        Self {
            inner: Mutex::new(ProfileDatabase::new()),
        }
    }

    #[pyo3(signature = (cache_key, region_id, resource, latency_s, workspace_bytes=0, status="measured", transfer_latency_s=0.0, io_latency_s=0.0))]
    fn insert(
        &self,
        cache_key: &str,
        region_id: &str,
        resource: &str,
        latency_s: f64,
        workspace_bytes: u64,
        status: &str,
        transfer_latency_s: f64,
        io_latency_s: f64,
    ) -> PyResult<()> {
        let status = parse_status(status)?;
        let rec = ProfileRecord {
            cache_key: cache_key.to_owned(),
            costs: vec![RegionCost {
                region_id: region_id.to_owned(),
                resource: resource.to_owned(),
                latency_s,
                workspace_bytes,
                status,
            }],
            transfer_latency_s,
            io_latency_s,
            status,
            notes: vec![],
        };
        self.inner
            .lock()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            .insert(rec);
        Ok(())
    }

    fn aggregate_median_latency(&self, cache_key: &str) -> PyResult<Option<f64>> {
        Ok(self
            .inner
            .lock()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            .aggregate_median_latency(cache_key))
    }

    fn save_json(&self, path: &str) -> PyResult<()> {
        self.inner
            .lock()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            .save_json(path)
            .map_err(PyRuntimeError::new_err)
    }

    fn load_json(&self, path: &str) -> PyResult<()> {
        self.inner
            .lock()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            .load_json(path)
            .map_err(PyRuntimeError::new_err)
    }

    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let g = self
            .inner
            .lock()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        // ProfileDatabase doesn't expose len; approximate via get of empty.
        let d = PyDict::new(py);
        d.set_item("native_profiler", true)?;
        let _ = g;
        Ok(d)
    }
}

fn parse_status(s: &str) -> PyResult<CostStatus> {
    match s.to_ascii_lowercase().as_str() {
        "measured" => Ok(CostStatus::Measured),
        "simulated" => Ok(CostStatus::Simulated),
        "estimated" => Ok(CostStatus::Estimated),
        "unknown" => Ok(CostStatus::Unknown),
        other => Err(PyValueError::new_err(format!("unknown cost status {other}"))),
    }
}
