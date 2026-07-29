//! Persistent native compiled artifact: schedule converted once, reused every forward.

use crate::{
    report_to_dict, schedule_from_py, SCHEDULE_FROM_PY_CALLS, SCHEDULER_ENTERS,
};
use parking_lot::Mutex;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use streamcompiler_core::{assert_schedule_valid, ExecutableSchedule};
use streamcompiler_runtime::{
    execute_schedule_ex, ExecuteOptions, InstructionCallback, InstructionCallbackResult,
    RegionCallback,
};

/// Shared cancellation flag owned by an execution context / compiled module.
#[pyclass(module = "streamcompiler._native", name = "NativeCancelToken")]
#[derive(Clone)]
pub struct NativeCancelToken {
    flag: Arc<AtomicBool>,
}

#[pymethods]
impl NativeCancelToken {
    #[new]
    fn new() -> Self {
        Self {
            flag: Arc::new(AtomicBool::new(false)),
        }
    }

    fn cancel(&self) {
        self.flag.store(true, Ordering::Release);
    }

    fn reset(&self) {
        self.flag.store(false, Ordering::Release);
    }

    fn is_cancelled(&self) -> bool {
        self.flag.load(Ordering::Acquire)
    }

    fn clone_token(&self) -> Self {
        Self {
            flag: Arc::clone(&self.flag),
        }
    }
}

impl NativeCancelToken {
    pub(crate) fn arc(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.flag)
    }
}

static ARTIFACT_ID_SEQ: AtomicU64 = AtomicU64::new(1);

/// Immutable native schedule handle. Created at specialize time; forwards reuse it.
#[pyclass(module = "streamcompiler._native", name = "NativeCompiledArtifact")]
pub struct NativeCompiledArtifact {
    schedule: Arc<ExecutableSchedule>,
    artifact_id: u64,
    /// Number of times this handle was used to execute (not convert).
    execute_count: AtomicU64,
    /// Serialize fingerprint after create for mutation checks.
    serialized: Mutex<Vec<u8>>,
}

#[pymethods]
impl NativeCompiledArtifact {
    /// Convert a Python ExecutableSchedule exactly once.
    #[staticmethod]
    fn from_schedule(schedule: &Bound<'_, PyAny>) -> PyResult<Self> {
        let s = schedule_from_py(schedule)?;
        assert_schedule_valid(&s).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let bytes = s
            .to_json_bytes()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            schedule: Arc::new(s),
            artifact_id: ARTIFACT_ID_SEQ.fetch_add(1, Ordering::Relaxed),
            execute_count: AtomicU64::new(0),
            serialized: Mutex::new(bytes),
        })
    }

    #[getter]
    fn artifact_id(&self) -> u64 {
        self.artifact_id
    }

    #[getter]
    fn fingerprint(&self) -> String {
        self.schedule.fingerprint.clone()
    }

    #[getter]
    fn graph_name(&self) -> String {
        self.schedule.graph_name.clone()
    }

    #[getter]
    fn instruction_count(&self) -> usize {
        self.schedule.instructions.len()
    }

    #[getter]
    fn execute_count(&self) -> u64 {
        self.execute_count.load(Ordering::Relaxed)
    }

    /// JSON bytes of the immutable schedule (identity check across runs).
    fn serialized_fingerprint(&self) -> Vec<u8> {
        self.serialized.lock().clone()
    }

    /// True when serialization matches the bytes captured at construction.
    fn is_unmutated(&self) -> bool {
        match self.schedule.to_json_bytes() {
            Ok(now) => *self.serialized.lock() == now,
            Err(_) => false,
        }
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        Ok(crate::schedule_to_dict(py, &self.schedule)?.into())
    }

    /// Execute without reconverting the schedule.
    #[pyo3(signature = (region_callback=None, instruction_handler=None, dry_run=false, cpu_workers=4, cancel_token=None))]
    fn execute(
        &self,
        py: Python<'_>,
        region_callback: Option<PyObject>,
        instruction_handler: Option<PyObject>,
        dry_run: bool,
        cpu_workers: usize,
        cancel_token: Option<&NativeCancelToken>,
    ) -> PyResult<PyObject> {
        SCHEDULER_ENTERS.fetch_add(1, Ordering::Relaxed);
        self.execute_count.fetch_add(1, Ordering::Relaxed);

        let opts = ExecuteOptions {
            dry_run_compute: dry_run && instruction_handler.is_none() && region_callback.is_none(),
            cpu_workers,
            ..Default::default()
        };
        let cb: Option<RegionCallback> = region_callback.map(|callable| {
            let callable = Arc::new(callable);
            Arc::new(move |region: &str, inputs: &[String], outputs: &[String]| {
                crate::INSTRUCTION_CALLBACKS.fetch_add(1, Ordering::Relaxed);
                Python::with_gil(|py| {
                    crate::GIL_ACQUISITIONS.fetch_add(1, Ordering::Relaxed);
                    callable
                        .call1(py, (region, inputs.to_vec(), outputs.to_vec()))
                        .map_err(|e| e.to_string())?;
                    Ok(())
                })
            }) as RegionCallback
        });
        let icb: Option<InstructionCallback> = instruction_handler.map(|callable| {
            let callable = Arc::new(callable);
            Arc::new(move |name: &str| {
                crate::INSTRUCTION_CALLBACKS.fetch_add(1, Ordering::Relaxed);
                Python::with_gil(|py| {
                    crate::GIL_ACQUISITIONS.fetch_add(1, Ordering::Relaxed);
                    let result = callable.call1(py, (name,)).map_err(|e| e.to_string())?;
                    if result.is_none(py) {
                        return Ok(InstructionCallbackResult::default());
                    }
                    let nbytes = result
                        .call_method1(py, "get", ("nbytes", 0))
                        .ok()
                        .and_then(|v| v.extract::<u64>(py).ok())
                        .unwrap_or(0);
                    let simulated = result
                        .call_method1(py, "get", ("simulated", false))
                        .ok()
                        .and_then(|v| v.extract::<bool>(py).ok())
                        .unwrap_or(false);
                    let notes = result
                        .call_method1(py, "get", ("notes", ""))
                        .ok()
                        .and_then(|v| v.extract::<String>(py).ok())
                        .unwrap_or_default();
                    Ok(InstructionCallbackResult {
                        nbytes,
                        simulated,
                        notes,
                    })
                })
            }) as InstructionCallback
        });

        let cancel = cancel_token
            .map(NativeCancelToken::arc)
            .unwrap_or_else(|| Arc::new(AtomicBool::new(false)));
        let schedule = Arc::clone(&self.schedule);
        let result = py
            .allow_threads(|| execute_schedule_ex(&schedule, &opts, cb, icb, Some(cancel)))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        report_to_dict(py, &result)
    }
}

/// Debug counters proving which native path was entered (test builds / diagnostics).
#[pyfunction]
pub fn debug_counters(py: Python<'_>) -> PyResult<PyObject> {
    let d = PyDict::new(py);
    d.set_item(
        "schedule_from_py_calls",
        SCHEDULE_FROM_PY_CALLS.load(Ordering::Relaxed),
    )?;
    d.set_item("scheduler_enters", SCHEDULER_ENTERS.load(Ordering::Relaxed))?;
    d.set_item(
        "instruction_callbacks",
        crate::INSTRUCTION_CALLBACKS.load(Ordering::Relaxed),
    )?;
    d.set_item(
        "gil_acquisitions",
        crate::GIL_ACQUISITIONS.load(Ordering::Relaxed),
    )?;
    d.set_item(
        "python_fallback_enters",
        crate::PYTHON_FALLBACK_ENTERS.load(Ordering::Relaxed),
    )?;
    Ok(d.into())
}

#[pyfunction]
pub fn reset_debug_counters() {
    SCHEDULE_FROM_PY_CALLS.store(0, Ordering::Relaxed);
    SCHEDULER_ENTERS.store(0, Ordering::Relaxed);
    crate::INSTRUCTION_CALLBACKS.store(0, Ordering::Relaxed);
    crate::GIL_ACQUISITIONS.store(0, Ordering::Relaxed);
    crate::PYTHON_FALLBACK_ENTERS.store(0, Ordering::Relaxed);
}

#[pyfunction]
pub fn record_python_fallback_enter() {
    crate::PYTHON_FALLBACK_ENTERS.fetch_add(1, Ordering::Relaxed);
}
