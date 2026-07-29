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
    #[allow(clippy::too_many_arguments)]
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
        let keys = g.get("__stats_probe__"); // empty unless inserted; used for lock liveness
        let d = PyDict::new(py);
        d.set_item("native_profiler", true)?;
        d.set_item("probe_records", keys.len())?;
        Ok(d)
    }

    fn get_region_median(&self, cache_key: &str, region_id: &str) -> PyResult<Option<f64>> {
        let recs = self
            .inner
            .lock()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            .get(cache_key);
        let mut vals: Vec<f64> = Vec::new();
        for r in &recs {
            for c in &r.costs {
                if c.region_id == region_id {
                    vals.push(c.latency_s);
                }
            }
        }
        if vals.is_empty() {
            return Ok(None);
        }
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mid = vals.len() / 2;
        Ok(Some(if vals.len() % 2 == 0 {
            (vals[mid - 1] + vals[mid]) / 2.0
        } else {
            vals[mid]
        }))
    }
}

fn parse_status(s: &str) -> PyResult<CostStatus> {
    match s.to_ascii_lowercase().as_str() {
        "measured" => Ok(CostStatus::Measured),
        "simulated" => Ok(CostStatus::Simulated),
        "estimated" => Ok(CostStatus::Estimated),
        "unknown" => Ok(CostStatus::Unknown),
        other => Err(PyValueError::new_err(format!(
            "unknown cost status {other}"
        ))),
    }
}
