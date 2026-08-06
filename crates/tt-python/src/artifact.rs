//! Persistent native compiled artifact: schedule converted once, reused every forward.

use crate::context_py::PyNativeExecutionContext;
use crate::{report_to_dict, schedule_from_py, SCHEDULER_ENTERS, SCHEDULE_FROM_PY_CALLS};
use parking_lot::Mutex;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use tt_ir::{assert_schedule_valid, ExecutableArtifact};
use tt_runtime::{
    execute_schedule_ex, execute_schedule_with_context, CopySyncCallback, ExecuteOptions,
    HandleReleaseCallback, InstructionCallback, InstructionCallbackResult, RegionCallback,
};

/// Shared cancellation flag owned by an execution context / compiled module.
// pyo3 0.29 makes the FromPyObject derive for Clone-able #[pyclass] types
// opt-in. This type is accepted as a function argument from Python, so opt in
// explicitly to preserve the existing conversion behaviour.
#[pyclass(
    module = "tensortorrent._native",
    name = "NativeCancelToken",
    from_py_object
)]
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
#[pyclass(module = "tensortorrent._native", name = "NativeCompiledArtifact")]
pub struct NativeCompiledArtifact {
    artifact: Arc<ExecutableArtifact>,
    artifact_id: u64,
    /// Number of times this handle was used to execute (not convert).
    execute_count: AtomicU64,
    /// Serialize fingerprint after create for mutation checks.
    serialized: Mutex<Vec<u8>>,
}

#[pymethods]
impl NativeCompiledArtifact {
    /// Convert a Python ExecutableSchedule exactly once into a versioned artifact.
    #[staticmethod]
    fn from_schedule(schedule: &Bound<'_, PyAny>) -> PyResult<Self> {
        let s = schedule_from_py(schedule)?;
        assert_schedule_valid(&s).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let art = ExecutableArtifact::from_schedule(s);
        art.validate()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let bytes = art
            .to_json_bytes()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        crate::NATIVE_ARTIFACT_CREATED.fetch_add(1, Ordering::Relaxed);
        Ok(Self {
            artifact: Arc::new(art),
            artifact_id: ARTIFACT_ID_SEQ.fetch_add(1, Ordering::Relaxed),
            execute_count: AtomicU64::new(0),
            serialized: Mutex::new(bytes),
        })
    }

    /// Load a versioned ExecutableArtifact from JSON bytes (never pickle).
    #[staticmethod]
    fn from_json_bytes(bytes: &[u8]) -> PyResult<Self> {
        let art = ExecutableArtifact::from_json_bytes(bytes)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let serialized = art
            .to_json_bytes()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        crate::NATIVE_ARTIFACT_CREATED.fetch_add(1, Ordering::Relaxed);
        Ok(Self {
            artifact: Arc::new(art),
            artifact_id: ARTIFACT_ID_SEQ.fetch_add(1, Ordering::Relaxed),
            execute_count: AtomicU64::new(0),
            serialized: Mutex::new(serialized),
        })
    }

    fn to_json_bytes(&self) -> PyResult<Py<PyAny>> {
        let bytes = self
            .artifact
            .to_json_bytes()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Python::attach(|py| Ok(PyBytes::new(py, &bytes).into()))
    }

    #[getter]
    fn format_version(&self) -> u32 {
        self.artifact.format_version
    }

    #[getter]
    fn compatibility_version(&self) -> String {
        self.artifact.compatibility_version.clone()
    }

    #[getter]
    fn graph_identity(&self) -> String {
        self.artifact.graph_identity.clone()
    }

    #[getter]
    fn artifact_id(&self) -> u64 {
        self.artifact_id
    }

    #[getter]
    fn fingerprint(&self) -> String {
        self.artifact.schedule.fingerprint.clone()
    }

    #[getter]
    fn graph_name(&self) -> String {
        self.artifact.schedule.graph_name.clone()
    }

    #[getter]
    fn instruction_count(&self) -> usize {
        self.artifact.schedule.instructions.len()
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
        match self.artifact.to_json_bytes() {
            Ok(now) => *self.serialized.lock() == now,
            Err(_) => false,
        }
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(crate::schedule_to_dict(py, &self.artifact.schedule)?.into())
    }

    /// Execute without reconverting the schedule.
    ///
    /// When ``execution_context`` is supplied, residency / events / allocations
    /// use that shared context (same store as ``NativeResidencySession.from_execution_context``).
    #[pyo3(signature = (region_callback=None, instruction_handler=None, dematerialize_callback=None, materialize_callback=None, parameter_load_callback=None, handle_release_callback=None, copy_sync_callback=None, dry_run=false, cpu_workers=4, cancel_token=None, execution_context=None))]
    #[allow(clippy::too_many_arguments)]
    fn execute(
        &self,
        py: Python<'_>,
        region_callback: Option<Py<PyAny>>,
        instruction_handler: Option<Py<PyAny>>,
        dematerialize_callback: Option<Py<PyAny>>,
        materialize_callback: Option<Py<PyAny>>,
        parameter_load_callback: Option<Py<PyAny>>,
        handle_release_callback: Option<Py<PyAny>>,
        copy_sync_callback: Option<Py<PyAny>>,
        dry_run: bool,
        cpu_workers: usize,
        cancel_token: Option<&NativeCancelToken>,
        execution_context: Option<&PyNativeExecutionContext>,
    ) -> PyResult<Py<PyAny>> {
        SCHEDULER_ENTERS.fetch_add(1, Ordering::Relaxed);
        self.execute_count.fetch_add(1, Ordering::Relaxed);

        let dematerialize = dematerialize_callback.map(|callable| {
            let callable = Arc::new(callable);
            Arc::new(move |tensor_id: &str| {
                Python::attach(|py| {
                    crate::GIL_ACQUISITIONS.fetch_add(1, Ordering::Relaxed);
                    crate::SPILL_DEMATERIALIZE_CALLBACKS.fetch_add(1, Ordering::Relaxed);
                    let result = callable
                        .call1(py, (tensor_id,))
                        .map_err(|e| e.to_string())?;
                    let dtype = result
                        .call_method1(py, "get", ("dtype", ""))
                        .map_err(|e| e.to_string())?
                        .extract::<String>(py)
                        .map_err(|e| e.to_string())?;
                    let shape = result
                        .call_method1(py, "get", ("shape", Vec::<i64>::new()))
                        .map_err(|e| e.to_string())?
                        .extract::<Vec<i64>>(py)
                        .map_err(|e| e.to_string())?;
                    let bytes = result
                        .call_method1(py, "get", ("bytes", b"".as_slice()))
                        .map_err(|e| e.to_string())?
                        .extract::<Vec<u8>>(py)
                        .map_err(|e| e.to_string())?;
                    Ok((
                        tt_storage::SpillMeta {
                            dtype,
                            shape,
                            nbytes: bytes.len() as u64,
                        },
                        bytes,
                    ))
                })
            }) as tt_runtime::DematerializeCallback
        });
        let materialize = materialize_callback.map(|callable| {
            let callable = Arc::new(callable);
            Arc::new(
                move |tensor_id: &str, meta: &tt_storage::SpillMeta, bytes: &[u8]| {
                    Python::attach(|py| {
                        crate::GIL_ACQUISITIONS.fetch_add(1, Ordering::Relaxed);
                        crate::SPILL_MATERIALIZE_CALLBACKS.fetch_add(1, Ordering::Relaxed);
                        let _ = callable
                            .call1(
                                py,
                                (
                                    tensor_id,
                                    meta.dtype.as_str(),
                                    meta.shape.clone(),
                                    PyBytes::new(py, bytes),
                                ),
                            )
                            .map_err(|e| e.to_string())?;
                        Ok(())
                    })
                },
            ) as tt_runtime::MaterializeCallback
        });
        let parameter_load = parameter_load_callback.map(|callable| {
            let callable = Arc::new(callable);
            Arc::new(move |pairs: &[(String, String)]| {
                Python::attach(|py| {
                    crate::GIL_ACQUISITIONS.fetch_add(1, Ordering::Relaxed);
                    crate::PARAMETER_LOAD_CALLBACKS.fetch_add(1, Ordering::Relaxed);
                    let batch = PyList::empty(py);
                    for (tid, dest) in pairs {
                        batch
                            .append((tid.as_str(), dest.as_str()))
                            .map_err(|e| e.to_string())?;
                    }
                    let result = callable.call1(py, (batch,)).map_err(|e| e.to_string())?;
                    if result.is_none(py) {
                        return Ok(vec![0u64; pairs.len()]);
                    }
                    result.extract::<Vec<u64>>(py).map_err(|e| e.to_string())
                })
            }) as tt_runtime::ParameterLoadCallback
        });
        let handle_release = handle_release_callback.map(|callable| {
            let callable = Arc::new(callable);
            Arc::new(move |pairs: &[(String, String)]| {
                Python::attach(|py| {
                    crate::GIL_ACQUISITIONS.fetch_add(1, Ordering::Relaxed);
                    crate::HANDLE_RELEASE_CALLBACKS.fetch_add(1, Ordering::Relaxed);
                    let batch = PyList::empty(py);
                    for (tid, rid) in pairs {
                        batch
                            .append((tid.as_str(), rid.as_str()))
                            .map_err(|e| e.to_string())?;
                    }
                    callable.call1(py, (batch,)).map_err(|e| e.to_string())?;
                    Ok(())
                })
            }) as HandleReleaseCallback
        });
        let copy_sync = copy_sync_callback.map(|callable| {
            let callable = Arc::new(callable);
            Arc::new(move |pairs: &[(String, String, String, u64)]| {
                Python::attach(|py| {
                    crate::GIL_ACQUISITIONS.fetch_add(1, Ordering::Relaxed);
                    crate::COPY_SYNC_CALLBACKS.fetch_add(1, Ordering::Relaxed);
                    let batch = PyList::empty(py);
                    for (tid, src, dst, n) in pairs {
                        batch
                            .append((tid.as_str(), src.as_str(), dst.as_str(), *n))
                            .map_err(|e| e.to_string())?;
                    }
                    callable.call1(py, (batch,)).map_err(|e| e.to_string())?;
                    Ok(())
                })
            }) as CopySyncCallback
        });

        let opts = ExecuteOptions {
            dry_run_compute: dry_run && instruction_handler.is_none() && region_callback.is_none(),
            cpu_workers,
            dematerialize,
            materialize,
            parameter_load,
            handle_release,
            copy_sync,
            ..Default::default()
        };
        let cb: Option<RegionCallback> = region_callback.map(|callable| {
            let callable = Arc::new(callable);
            Arc::new(move |invocations: &[tt_runtime::RegionInvocation]| {
                crate::INSTRUCTION_CALLBACKS.fetch_add(1, Ordering::Relaxed);
                crate::COMPUTE_CALLBACKS.fetch_add(1, Ordering::Relaxed);
                Python::attach(|py| {
                    crate::GIL_ACQUISITIONS.fetch_add(1, Ordering::Relaxed);
                    let batch = PyList::empty(py);
                    for inv in invocations {
                        batch
                            .append((
                                inv.region_id.as_str(),
                                inv.inputs.clone(),
                                inv.outputs.clone(),
                            ))
                            .map_err(|e| e.to_string())?;
                    }
                    callable.call1(py, (batch,)).map_err(|e| e.to_string())?;
                    Ok(())
                })
            }) as RegionCallback
        });
        let icb: Option<InstructionCallback> = instruction_handler.map(|callable| {
            let callable = Arc::new(callable);
            Arc::new(move |name: &str| {
                crate::INSTRUCTION_CALLBACKS.fetch_add(1, Ordering::Relaxed);
                crate::NON_COMPUTE_PYTHON_CALLBACKS.fetch_add(1, Ordering::Relaxed);
                Python::attach(|py| {
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

        let schedule = Arc::new(self.artifact.schedule.clone());
        let result = if let Some(ectx) = execution_context {
            let ctx = Arc::clone(ectx.inner());
            py.detach(|| execute_schedule_with_context(&schedule, &opts, cb, icb, ctx))
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
        } else {
            let cancel = cancel_token
                .map(NativeCancelToken::arc)
                .unwrap_or_else(|| Arc::new(AtomicBool::new(false)));
            py.detach(|| execute_schedule_ex(&schedule, &opts, cb, icb, Some(cancel)))
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
        };
        report_to_dict(py, &result)
    }
}

/// Debug counters proving which native path was entered (test builds / diagnostics).
#[pyfunction]
pub fn debug_counters(py: Python<'_>) -> PyResult<Py<PyAny>> {
    let d = PyDict::new(py);
    d.set_item(
        "schedule_from_py_calls",
        SCHEDULE_FROM_PY_CALLS.load(Ordering::Relaxed),
    )?;
    d.set_item(
        "schedule_conversions_during_forward",
        SCHEDULE_FROM_PY_CALLS.load(Ordering::Relaxed),
    )?;
    d.set_item("scheduler_enters", SCHEDULER_ENTERS.load(Ordering::Relaxed))?;
    d.set_item(
        "native_scheduler_entries",
        SCHEDULER_ENTERS.load(Ordering::Relaxed),
    )?;
    d.set_item(
        "instruction_callbacks",
        crate::INSTRUCTION_CALLBACKS.load(Ordering::Relaxed),
    )?;
    d.set_item(
        "compute_callbacks",
        crate::COMPUTE_CALLBACKS.load(Ordering::Relaxed),
    )?;
    d.set_item(
        "non_compute_python_callbacks",
        crate::NON_COMPUTE_PYTHON_CALLBACKS.load(Ordering::Relaxed),
    )?;
    d.set_item(
        "gil_acquisitions",
        crate::GIL_ACQUISITIONS.load(Ordering::Relaxed),
    )?;
    d.set_item(
        "parameter_load_callbacks",
        crate::PARAMETER_LOAD_CALLBACKS.load(Ordering::Relaxed),
    )?;
    d.set_item(
        "parameter_release_callbacks",
        crate::PARAMETER_RELEASE_CALLBACKS.load(Ordering::Relaxed),
    )?;
    d.set_item(
        "spill_dematerialize_callbacks",
        crate::SPILL_DEMATERIALIZE_CALLBACKS.load(Ordering::Relaxed),
    )?;
    d.set_item(
        "spill_materialize_callbacks",
        crate::SPILL_MATERIALIZE_CALLBACKS.load(Ordering::Relaxed),
    )?;
    d.set_item(
        "handle_release_callbacks",
        crate::HANDLE_RELEASE_CALLBACKS.load(Ordering::Relaxed),
    )?;
    d.set_item(
        "copy_sync_callbacks",
        crate::COPY_SYNC_CALLBACKS.load(Ordering::Relaxed),
    )?;
    d.set_item(
        "python_fallback_enters",
        crate::PYTHON_FALLBACK_ENTERS.load(Ordering::Relaxed),
    )?;
    d.set_item(
        "native_artifact_created",
        crate::NATIVE_ARTIFACT_CREATED.load(Ordering::Relaxed),
    )?;
    Ok(d.into())
}

#[pyfunction]
pub fn reset_debug_counters() {
    SCHEDULE_FROM_PY_CALLS.store(0, Ordering::Relaxed);
    SCHEDULER_ENTERS.store(0, Ordering::Relaxed);
    crate::INSTRUCTION_CALLBACKS.store(0, Ordering::Relaxed);
    crate::COMPUTE_CALLBACKS.store(0, Ordering::Relaxed);
    crate::NON_COMPUTE_PYTHON_CALLBACKS.store(0, Ordering::Relaxed);
    crate::GIL_ACQUISITIONS.store(0, Ordering::Relaxed);
    crate::PARAMETER_LOAD_CALLBACKS.store(0, Ordering::Relaxed);
    crate::PARAMETER_RELEASE_CALLBACKS.store(0, Ordering::Relaxed);
    crate::SPILL_DEMATERIALIZE_CALLBACKS.store(0, Ordering::Relaxed);
    crate::SPILL_MATERIALIZE_CALLBACKS.store(0, Ordering::Relaxed);
    crate::HANDLE_RELEASE_CALLBACKS.store(0, Ordering::Relaxed);
    crate::COPY_SYNC_CALLBACKS.store(0, Ordering::Relaxed);
    crate::PYTHON_FALLBACK_ENTERS.store(0, Ordering::Relaxed);
    crate::NATIVE_ARTIFACT_CREATED.store(0, Ordering::Relaxed);
}

#[pyfunction]
pub fn record_parameter_release() {
    crate::PARAMETER_RELEASE_CALLBACKS.fetch_add(1, Ordering::Relaxed);
}
