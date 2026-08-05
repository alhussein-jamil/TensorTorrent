//! Schedule simulation and execution Python bindings.

use crate::counters::{
    COMPUTE_CALLBACKS, GIL_ACQUISITIONS, INSTRUCTION_CALLBACKS, NON_COMPUTE_PYTHON_CALLBACKS,
    SCHEDULER_ENTERS,
};
use crate::machine_py::machine_from_py;
use crate::schedule_py::schedule_from_py;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tt_ir::ExecutableSchedule;
use tt_runtime::{
    execute_schedule_ex, simulate_schedule, ExecuteOptions, ExecuteReport, InstructionCallback,
    InstructionCallbackResult, RegionCallback, SimulationOutcome,
};

#[pyfunction]
#[pyo3(signature = (schedule, machine=None))]
pub(crate) fn simulate_schedule_py(
    py: Python<'_>,
    schedule: &Bound<'_, PyAny>,
    machine: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let s = schedule_from_py(schedule)?;
    let m = machine_from_py(machine)?;
    let outcome = py
        .detach(|| simulate_schedule(&s, &m))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    match outcome {
        SimulationOutcome::Valid(result) => simulation_result_to_dict(py, &result),
        SimulationOutcome::InfeasibleMemory(rep) => {
            let d = PyDict::new(py);
            d.set_item("status", "infeasible_memory")?;
            d.set_item("simulated", true)?;
            d.set_item("memory", &rep.memory)?;
            d.set_item("resident_bytes", rep.resident_bytes)?;
            d.set_item("allocatable_bytes", rep.allocatable_bytes)?;
            d.set_item("instruction", &rep.instruction)?;
            d.set_item("at_s", rep.at_s)?;
            d.set_item("peak_bytes", &rep.peak_bytes)?;
            Err(PyValueError::new_err(format!(
                "schedule infeasible: memory {} resident={} allocatable={} at instruction {:?}",
                rep.memory, rep.resident_bytes, rep.allocatable_bytes, rep.instruction
            )))
        }
        SimulationOutcome::InvalidResidency { detail }
        | SimulationOutcome::InvalidEvent { detail }
        | SimulationOutcome::Unsupported { detail } => Err(PyValueError::new_err(detail)),
    }
}

fn simulation_result_to_dict(
    py: Python<'_>,
    result: &tt_runtime::SimulationResult,
) -> PyResult<Py<PyAny>> {
    let d = PyDict::new(py);
    d.set_item("status", "valid")?;
    d.set_item("makespan_s", result.makespan_s)?;
    d.set_item("peak_bytes", &result.peak_bytes)?;
    d.set_item(
        "exposed_transfer_latency_s",
        result.exposed_transfer_latency_s,
    )?;
    d.set_item("resource_busy_s", &result.resource_busy_s)?;
    d.set_item("simulated", true)?;
    d.set_item("critical_path", &result.critical_path)?;
    d.set_item("bytes_read", result.bytes_read)?;
    d.set_item("bytes_transferred", result.bytes_transferred)?;
    d.set_item("instruction_count", result.instruction_count)?;
    d.set_item("activation_peak_bytes", result.activation_peak_bytes)?;
    let timeline = PyList::empty(py);
    for ev in &result.timeline {
        let e = PyDict::new(py);
        e.set_item("name", &ev.name)?;
        e.set_item("instruction", &ev.name)?;
        e.set_item("opcode", &ev.opcode)?;
        e.set_item("resource", &ev.resource)?;
        e.set_item("start_s", ev.start_s)?;
        e.set_item("end_s", ev.end_s)?;
        e.set_item("nbytes", ev.nbytes)?;
        e.set_item("simulated", true)?;
        e.set_item("critical_pred", &ev.critical_pred)?;
        if let Some(ref event) = ev.event {
            e.set_item("event", event)?;
        } else {
            e.set_item("event", &ev.opcode)?;
        }
        if let Some(ref memory) = ev.memory {
            e.set_item("memory", memory)?;
        }
        if let Some(rb) = ev.resident_bytes {
            e.set_item("resident_bytes", rb)?;
        }
        if let Some(ab) = ev.allocatable_bytes {
            e.set_item("allocatable_bytes", ab)?;
        }
        if let Some(at) = ev.at_s {
            e.set_item("at_s", at)?;
        }
        if let Some(written) = ev.activation_bytes_written {
            e.set_item("activation_bytes_written", written)?;
        }
        timeline.append(e)?;
    }
    d.set_item("timeline", timeline)?;
    // transfer_events / release_events as JSON-compatible dict lists
    let transfers = PyList::empty(py);
    for te in &result.transfer_events {
        let obj: Py<PyAny> = pythonize_json(py, te)?;
        transfers.append(obj)?;
    }
    d.set_item("transfer_events", transfers)?;
    let releases = PyList::empty(py);
    for re in &result.release_events {
        let obj: Py<PyAny> = pythonize_json(py, re)?;
        releases.append(obj)?;
    }
    d.set_item("release_events", releases)?;
    Ok(d.into())
}

fn pythonize_json(py: Python<'_>, value: &serde_json::Value) -> PyResult<Py<PyAny>> {
    match value {
        serde_json::Value::Null => Ok(py.None()),
        serde_json::Value::Bool(b) => Ok((*b).into_pyobject(py)?.to_owned().into_any().unbind()),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(i.into_pyobject(py)?.to_owned().into_any().unbind())
            } else if let Some(u) = n.as_u64() {
                Ok(u.into_pyobject(py)?.to_owned().into_any().unbind())
            } else {
                Ok(n.as_f64()
                    .unwrap_or(0.0)
                    .into_pyobject(py)?
                    .to_owned()
                    .into_any()
                    .unbind())
            }
        }
        serde_json::Value::String(s) => {
            Ok(s.as_str().into_pyobject(py)?.to_owned().into_any().unbind())
        }
        serde_json::Value::Array(arr) => {
            let list = PyList::empty(py);
            for item in arr {
                list.append(pythonize_json(py, item)?)?;
            }
            Ok(list.unbind().into())
        }
        serde_json::Value::Object(map) => {
            let d = PyDict::new(py);
            for (k, v) in map {
                d.set_item(k, pythonize_json(py, v)?)?;
            }
            Ok(d.unbind().into())
        }
    }
}

pub(crate) fn report_to_dict(py: Python<'_>, result: &ExecuteReport) -> PyResult<Py<PyAny>> {
    let d = PyDict::new(py);
    d.set_item("wall_time_s", result.wall_time_s)?;
    d.set_item("peak_activation_bytes", result.peak_activation_bytes)?;
    d.set_item("allocation_peak_bytes", result.allocation_peak_bytes)?;
    d.set_item("bytes_read", result.bytes_read)?;
    d.set_item("bytes_transferred", result.bytes_transferred)?;
    d.set_item("simulated_ops", result.simulated_ops)?;
    let intervals: Vec<(f64, f64)> = result
        .events
        .iter()
        .map(|ev| (ev.start_s, ev.end_s))
        .collect();
    d.set_item(
        "max_concurrent",
        tt_runtime::max_concurrency_from_intervals(&intervals),
    )?;
    let events = PyList::empty(py);
    for ev in &result.events {
        let e = PyDict::new(py);
        e.set_item("name", &ev.name)?;
        e.set_item("opcode", &ev.opcode)?;
        e.set_item("resource", &ev.resource)?;
        e.set_item("submitted_s", ev.submitted_s)?;
        e.set_item("start_s", ev.start_s)?;
        e.set_item("end_s", ev.end_s)?;
        e.set_item("nbytes", ev.nbytes)?;
        e.set_item("simulated", ev.simulated)?;
        e.set_item("notes", &ev.notes)?;
        events.append(e)?;
    }
    d.set_item("events", events)?;
    Ok(d.into())
}

#[pyfunction]
#[pyo3(signature = (schedule, region_callback=None, instruction_handler=None, dry_run=false, cpu_workers=4))]
pub(crate) fn execute_schedule_py(
    py: Python<'_>,
    schedule: &Bound<'_, PyAny>,
    region_callback: Option<Py<PyAny>>,
    instruction_handler: Option<Py<PyAny>>,
    dry_run: bool,
    cpu_workers: usize,
) -> PyResult<Py<PyAny>> {
    SCHEDULER_ENTERS.fetch_add(1, Ordering::Relaxed);
    let s = schedule_from_py(schedule)?;
    let opts = ExecuteOptions {
        dry_run_compute: dry_run && instruction_handler.is_none() && region_callback.is_none(),
        cpu_workers,
        ..Default::default()
    };
    let cb: Option<RegionCallback> = region_callback.map(|callable| {
        let callable = Arc::new(callable);
        Arc::new(move |invocations: &[tt_runtime::RegionInvocation]| {
            INSTRUCTION_CALLBACKS.fetch_add(1, Ordering::Relaxed);
            COMPUTE_CALLBACKS.fetch_add(1, Ordering::Relaxed);
            Python::attach(|py| {
                GIL_ACQUISITIONS.fetch_add(1, Ordering::Relaxed);
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
            INSTRUCTION_CALLBACKS.fetch_add(1, Ordering::Relaxed);
            NON_COMPUTE_PYTHON_CALLBACKS.fetch_add(1, Ordering::Relaxed);
            Python::attach(|py| {
                GIL_ACQUISITIONS.fetch_add(1, Ordering::Relaxed);
                let result = callable.call1(py, (name,)).map_err(|e| e.to_string())?;
                if result.is_none(py) {
                    return Ok(InstructionCallbackResult::default());
                }
                let nbytes = result
                    .getattr(py, "get")
                    .ok()
                    .and_then(|_| result.call_method1(py, "get", ("nbytes", 0)).ok())
                    .and_then(|v| v.extract::<u64>(py).ok())
                    .or_else(|| {
                        result
                            .getattr(py, "nbytes")
                            .ok()
                            .and_then(|v| v.extract(py).ok())
                    })
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

    let cancel = Arc::new(AtomicBool::new(false));
    let result = py
        .detach(|| execute_schedule_ex(&s, &opts, cb, icb, Some(cancel)))
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    report_to_dict(py, &result)
}

#[pyfunction]
pub(crate) fn execute_schedule_json(
    py: Python<'_>,
    json: &str,
    dry_run: bool,
) -> PyResult<Py<PyAny>> {
    let s =
        ExecutableSchedule::from_json(json).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let opts = ExecuteOptions {
        dry_run_compute: dry_run,
        ..Default::default()
    };
    let cancel = Arc::new(AtomicBool::new(false));
    let result = py
        .detach(|| execute_schedule_ex(&s, &opts, None, None, Some(cancel)))
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let d = PyDict::new(py);
    d.set_item("wall_time_s", result.wall_time_s)?;
    d.set_item("instruction_count", result.events.len())?;
    Ok(d.into())
}
