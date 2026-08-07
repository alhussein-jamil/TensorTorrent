//! PyO3 bindings: schedule round-trip, simulate, execute (GIL released).

mod artifact;
mod context_py;
mod counters;
mod cpu_backend_py;
mod execute_py;
mod machine_py;
mod planner_py;
mod profiler_py;
mod residency_py;
mod schedule_py;
mod storage_py;
mod virtual_backend_py;

pub(crate) use artifact::{
    debug_counters, record_parameter_release, reset_debug_counters, NativeCancelToken,
    NativeCompiledArtifact,
};
pub(crate) use context_py::PyNativeExecutionContext;
pub(crate) use counters::{
    COMPUTE_CALLBACKS, COPY_SYNC_CALLBACKS, GIL_ACQUISITIONS, HANDLE_RELEASE_CALLBACKS,
    INSTRUCTION_CALLBACKS, NATIVE_ARTIFACT_CREATED, NON_COMPUTE_PYTHON_CALLBACKS,
    PARAMETER_LOAD_CALLBACKS, PARAMETER_RELEASE_CALLBACKS, PYTHON_FALLBACK_ENTERS,
    SCHEDULER_ENTERS, SCHEDULE_FROM_PY_CALLS, SPILL_DEMATERIALIZE_CALLBACKS,
    SPILL_MATERIALIZE_CALLBACKS,
};
pub(crate) use cpu_backend_py::NativeCpuBackend;
pub(crate) use execute_py::report_to_dict;
pub(crate) use profiler_py::NativeProfileDatabase;
pub(crate) use residency_py::{new_native_residency, NativeResidencySession};
pub(crate) use schedule_py::{schedule_from_py, schedule_to_dict};
pub(crate) use storage_py::{
    read_activation_spill, remove_activation_spill, write_activation_spill, NativeChunkCache,
    NativePackReader, NativeStreamingStore,
};
pub(crate) use virtual_backend_py::{virtual_backend_pending_is_async, NativeVirtualBackend};

use execute_py::{
    execute_schedule_json, execute_schedule_py, simulate_schedule_py, simulate_schedules_py,
};
use planner_py::plan_placements_py;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use schedule_py::{
    assert_schedule_valid_py, schedule_from_json, schedule_roundtrip, schedule_to_json,
    validate_schedule_py,
};

#[pyfunction]
fn native_available() -> bool {
    true
}

#[pyfunction]
fn native_version() -> String {
    env!("CARGO_PKG_VERSION").to_owned()
}

/// Remove spill session directories left behind by dead processes.
///
/// Returns the number of orphaned `tt-spill-<pid>-<exec>` directories removed.
/// Live processes' sessions (including this one's) are never touched.
#[pyfunction]
fn sweep_orphan_spill_sessions(dir: &str) -> usize {
    tt_storage::sweep_orphan_spill_sessions(std::path::Path::new(dir))
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(native_available, m)?)?;
    m.add_function(wrap_pyfunction!(native_version, m)?)?;
    m.add_function(wrap_pyfunction!(sweep_orphan_spill_sessions, m)?)?;
    m.add_function(wrap_pyfunction!(schedule_to_json, m)?)?;
    m.add_function(wrap_pyfunction!(schedule_from_json, m)?)?;
    m.add_function(wrap_pyfunction!(schedule_roundtrip, m)?)?;
    m.add_function(wrap_pyfunction!(validate_schedule_py, m)?)?;
    m.add_function(wrap_pyfunction!(assert_schedule_valid_py, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_schedule_py, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_schedules_py, m)?)?;
    m.add_function(wrap_pyfunction!(plan_placements_py, m)?)?;
    m.add_function(wrap_pyfunction!(execute_schedule_py, m)?)?;
    m.add_function(wrap_pyfunction!(execute_schedule_json, m)?)?;
    m.add_function(wrap_pyfunction!(debug_counters, m)?)?;
    m.add_function(wrap_pyfunction!(reset_debug_counters, m)?)?;
    m.add_function(wrap_pyfunction!(record_parameter_release, m)?)?;
    m.add_function(wrap_pyfunction!(new_native_residency, m)?)?;
    m.add_class::<NativeCompiledArtifact>()?;
    m.add_class::<NativeCancelToken>()?;
    m.add_class::<PyNativeExecutionContext>()?;
    m.add_class::<NativeResidencySession>()?;
    m.add_class::<NativePackReader>()?;
    m.add_class::<NativeChunkCache>()?;
    m.add_class::<NativeStreamingStore>()?;
    m.add_class::<NativeProfileDatabase>()?;
    m.add_class::<NativeVirtualBackend>()?;
    m.add_class::<NativeCpuBackend>()?;
    m.add_function(wrap_pyfunction!(virtual_backend_pending_is_async, m)?)?;
    m.add_function(wrap_pyfunction!(write_activation_spill, m)?)?;
    m.add_function(wrap_pyfunction!(read_activation_spill, m)?)?;
    m.add_function(wrap_pyfunction!(remove_activation_spill, m)?)?;
    // Aliases without _py suffix for cleaner Python imports.
    m.add("validate_schedule", m.getattr("validate_schedule_py")?)?;
    m.add(
        "assert_schedule_valid",
        m.getattr("assert_schedule_valid_py")?,
    )?;
    m.add("simulate_schedule", m.getattr("simulate_schedule_py")?)?;
    m.add("simulate_schedules", m.getattr("simulate_schedules_py")?)?;
    m.add("plan_placements", m.getattr("plan_placements_py")?)?;
    m.add("execute_schedule", m.getattr("execute_schedule_py")?)?;
    Ok(())
}
