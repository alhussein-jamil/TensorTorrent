//! PyO3 bindings: schedule round-trip, simulate, execute (GIL released).

mod artifact;
mod context_py;
mod cpu_backend_py;
mod profiler_py;
mod residency_py;
mod storage_py;
mod virtual_backend_py;

pub(crate) use artifact::{
    debug_counters, record_parameter_release, reset_debug_counters, NativeCancelToken,
    NativeCompiledArtifact,
};
pub(crate) use context_py::PyNativeExecutionContext;
pub(crate) use cpu_backend_py::NativeCpuBackend;
pub(crate) use profiler_py::NativeProfileDatabase;
pub(crate) use residency_py::{new_native_residency, NativeResidencySession};
pub(crate) use storage_py::{
    read_activation_spill, remove_activation_spill, write_activation_spill, NativeChunkCache,
    NativePackReader, NativeStreamingStore,
};
pub(crate) use virtual_backend_py::{virtual_backend_pending_is_async, NativeVirtualBackend};

use indexmap::IndexMap;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList, PyModule};
use std::collections::BTreeMap;
use std::str::FromStr;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use tt_ir::{
    assert_schedule_valid, validate_schedule, AttrValue, ExecutableSchedule, Instruction,
    InstructionId, MemoryTier, Opcode, RegionId, ResourceId, StreamId, TensorId,
};
use tt_runtime::{
    execute_schedule_ex, simulate_schedule, ExecuteOptions, ExecuteReport, InstructionCallback,
    InstructionCallbackResult, MachineModel, MemoryResource, RegionCallback, SimulationOutcome,
    TransferLink,
};

pub(crate) static SCHEDULE_FROM_PY_CALLS: AtomicU64 = AtomicU64::new(0);
pub(crate) static SCHEDULER_ENTERS: AtomicU64 = AtomicU64::new(0);
pub(crate) static INSTRUCTION_CALLBACKS: AtomicU64 = AtomicU64::new(0);
pub(crate) static COMPUTE_CALLBACKS: AtomicU64 = AtomicU64::new(0);
pub(crate) static NON_COMPUTE_PYTHON_CALLBACKS: AtomicU64 = AtomicU64::new(0);
pub(crate) static GIL_ACQUISITIONS: AtomicU64 = AtomicU64::new(0);
pub(crate) static PARAMETER_LOAD_CALLBACKS: AtomicU64 = AtomicU64::new(0);
pub(crate) static PARAMETER_RELEASE_CALLBACKS: AtomicU64 = AtomicU64::new(0);
pub(crate) static SPILL_DEMATERIALIZE_CALLBACKS: AtomicU64 = AtomicU64::new(0);
pub(crate) static SPILL_MATERIALIZE_CALLBACKS: AtomicU64 = AtomicU64::new(0);
pub(crate) static HANDLE_RELEASE_CALLBACKS: AtomicU64 = AtomicU64::new(0);
pub(crate) static COPY_SYNC_CALLBACKS: AtomicU64 = AtomicU64::new(0);
pub(crate) static PYTHON_FALLBACK_ENTERS: AtomicU64 = AtomicU64::new(0);
pub(crate) static NATIVE_ARTIFACT_CREATED: AtomicU64 = AtomicU64::new(0);

fn attr_from_py(obj: &Bound<'_, PyAny>) -> PyResult<AttrValue> {
    if obj.is_none() {
        return Ok(AttrValue::Null);
    }
    if let Ok(v) = obj.extract::<bool>() {
        return Ok(AttrValue::Bool(v));
    }
    if let Ok(v) = obj.extract::<i64>() {
        return Ok(AttrValue::Int(v));
    }
    if let Ok(v) = obj.extract::<f64>() {
        // Prefer int when whole number? Keep float.
        return Ok(AttrValue::Float(v));
    }
    if let Ok(v) = obj.extract::<String>() {
        return Ok(AttrValue::String(v));
    }
    if let Ok(dict) = obj.cast::<PyDict>() {
        // Try int map first.
        let mut all_int = true;
        let mut int_map = BTreeMap::new();
        let mut any_map = BTreeMap::new();
        for (k, v) in dict.iter() {
            let key: String = k.extract()?;
            if let Ok(i) = v.extract::<i64>() {
                int_map.insert(key.clone(), i);
                any_map.insert(key, AttrValue::Int(i));
            } else {
                all_int = false;
                any_map.insert(key, attr_from_py(&v)?);
            }
        }
        if all_int {
            return Ok(AttrValue::IntMap(int_map));
        }
        return Ok(AttrValue::Map(any_map));
    }
    if let Ok(list) = obj.cast::<PyList>() {
        let mut out = Vec::with_capacity(list.len());
        for item in list.iter() {
            out.push(attr_from_py(&item)?);
        }
        return Ok(AttrValue::List(out));
    }
    Ok(AttrValue::String(obj.str()?.to_string()))
}

fn attr_to_py(py: Python<'_>, v: &AttrValue) -> PyResult<Py<PyAny>> {
    match v {
        AttrValue::Null => Ok(py.None()),
        AttrValue::Bool(b) => Ok((*b).into_pyobject(py)?.to_owned().into_any().unbind()),
        AttrValue::Int(i) => Ok((*i).into_pyobject(py)?.to_owned().into_any().unbind()),
        AttrValue::Float(f) => Ok((*f).into_pyobject(py)?.to_owned().into_any().unbind()),
        AttrValue::String(s) => Ok(s.as_str().into_pyobject(py)?.to_owned().into_any().unbind()),
        AttrValue::IntMap(m) => {
            let d = PyDict::new(py);
            for (k, v) in m {
                d.set_item(k, *v)?;
            }
            Ok(d.unbind().into())
        }
        AttrValue::StringMap(m) => {
            let d = PyDict::new(py);
            for (k, v) in m {
                d.set_item(k, v)?;
            }
            Ok(d.unbind().into())
        }
        AttrValue::List(xs) => {
            let list = PyList::empty(py);
            for x in xs {
                list.append(attr_to_py(py, x)?)?;
            }
            Ok(list.unbind().into())
        }
        AttrValue::Map(m) => {
            let d = PyDict::new(py);
            for (k, v) in m {
                d.set_item(k, attr_to_py(py, v)?)?;
            }
            Ok(d.unbind().into())
        }
    }
}

fn py_string_seq(obj: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    if obj.is_none() {
        return Ok(vec![]);
    }
    if let Ok(v) = obj.extract::<Vec<String>>() {
        return Ok(v);
    }
    let mut out = Vec::new();
    for item in obj.try_iter()? {
        out.push(item?.extract::<String>()?);
    }
    Ok(out)
}

fn instruction_from_py(obj: &Bound<'_, PyAny>) -> PyResult<Instruction> {
    let opcode_s: String = obj.getattr("opcode")?.extract::<String>().or_else(|_| {
        let op = obj.getattr("opcode")?;
        if let Ok(v) = op.getattr("value") {
            v.extract::<String>()
        } else {
            op.str()?.extract()
        }
    })?;
    let opcode = Opcode::from_str(&opcode_s).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let name: String = obj.getattr("name")?.extract()?;
    let resource: String = obj.getattr("resource")?.extract()?;
    let depends_on = py_string_seq(&obj.getattr("depends_on")?)?;
    let inputs = py_string_seq(&obj.getattr("inputs")?)?;
    let outputs = py_string_seq(&obj.getattr("outputs")?)?;
    let raw_nbytes: i64 = obj.getattr("nbytes")?.extract()?;
    let nbytes = u64::try_from(raw_nbytes)
        .map_err(|_| PyValueError::new_err("instruction nbytes must be non-negative"))?;
    let tier_s: String = obj
        .getattr("memory_tier")
        .ok()
        .and_then(|t| {
            t.extract::<String>()
                .or_else(|_| t.getattr("value").and_then(|v| v.extract()))
                .ok()
        })
        .unwrap_or_else(|| "unknown".into());
    let memory_tier = MemoryTier::from_str(&tier_s).unwrap_or(MemoryTier::Unknown);
    let predicted_duration_s: f64 = obj.getattr("predicted_duration_s")?.extract()?;
    if !predicted_duration_s.is_finite() || predicted_duration_s < 0.0 {
        return Err(PyValueError::new_err(
            "instruction predicted_duration_s must be finite and non-negative",
        ));
    }
    let executable_ref: Option<String> = obj
        .getattr("executable_ref")
        .ok()
        .and_then(|v| if v.is_none() { None } else { v.extract().ok() })
        .or_else(|| {
            if opcode == Opcode::Compute {
                Some(name.clone())
            } else {
                None
            }
        });
    let source: Option<String> =
        obj.getattr("source")
            .ok()
            .and_then(|v| if v.is_none() { None } else { v.extract().ok() });
    let destination: Option<String> =
        obj.getattr("destination").ok().and_then(
            |v| {
                if v.is_none() {
                    None
                } else {
                    v.extract().ok()
                }
            },
        );
    let backend_id: Option<String> =
        obj.getattr("backend_id").ok().and_then(
            |v| {
                if v.is_none() {
                    None
                } else {
                    v.extract().ok()
                }
            },
        );
    let transfer_backend: Option<String> = obj.getattr("transfer_backend").ok().and_then(|v| {
        if v.is_none() {
            None
        } else {
            v.extract().ok()
        }
    });
    let sync_required: bool = obj
        .getattr("sync_required")
        .ok()
        .and_then(|v| v.extract().ok())
        .unwrap_or(false);
    let stream_id: Option<String> =
        obj.getattr("stream_id").ok().and_then(
            |v| {
                if v.is_none() {
                    None
                } else {
                    v.extract().ok()
                }
            },
        );
    let copy_engine_id: Option<String> = obj.getattr("copy_engine_id").ok().and_then(|v| {
        if v.is_none() {
            None
        } else {
            v.extract().ok()
        }
    });
    let link_id: Option<String> =
        obj.getattr("link_id")
            .ok()
            .and_then(|v| if v.is_none() { None } else { v.extract().ok() });
    let io_queue_id: Option<String> =
        obj.getattr("io_queue_id").ok().and_then(
            |v| {
                if v.is_none() {
                    None
                } else {
                    v.extract().ok()
                }
            },
        );
    // Prefer explicit fields; fall back to attributes; then opcode defaults.
    let stream_from_attr = attributes_get_str(obj, "stream_id");
    let engine_from_attr = attributes_get_str(obj, "copy_engine_id");
    let link_from_attr = attributes_get_str(obj, "link_id");
    let io_from_attr = attributes_get_str(obj, "io_queue_id");
    let mut attributes = IndexMap::new();
    if let Ok(attrs) = obj.getattr("attributes") {
        if let Ok(dict) = attrs.cast::<PyDict>() {
            for (k, v) in dict.iter() {
                let key: String = k.extract()?;
                attributes.insert(key, attr_from_py(&v)?);
            }
        } else if attrs.hasattr("items")? {
            let items = attrs.call_method0("items")?;
            for item in items.try_iter()? {
                let item = item?;
                let key = item.get_item(0)?.extract::<String>()?;
                let val = item.get_item(1)?;
                attributes.insert(key, attr_from_py(&val)?);
            }
        } else if attrs.hasattr("as_dict")? {
            let dict = attrs.call_method0("as_dict")?;
            if let Ok(d) = dict.cast::<PyDict>() {
                for (k, v) in d.iter() {
                    let key: String = k.extract()?;
                    attributes.insert(key, attr_from_py(&v)?);
                }
            }
        }
    }
    let stream_id = stream_id
        .or(stream_from_attr)
        .or_else(|| default_stream_id(opcode, &resource));
    let copy_engine_id = copy_engine_id.or(engine_from_attr).or_else(|| {
        matches!(opcode, Opcode::Transfer | Opcode::Load | Opcode::Prefetch)
            .then(|| format!("{resource}::copy0"))
    });
    let link_id = link_id.or(link_from_attr).or_else(|| {
        (opcode == Opcode::Transfer).then(|| {
            let src = source.as_deref().unwrap_or("unknown");
            let dst = destination.as_deref().unwrap_or(&resource);
            format!("{src}->{dst}")
        })
    });
    let io_queue_id = io_queue_id.or(io_from_attr).or_else(|| {
        matches!(opcode, Opcode::Prefetch | Opcode::Load).then(|| format!("{resource}::io0"))
    });
    Ok(Instruction {
        opcode,
        name: InstructionId::new(name),
        resource: ResourceId::new(resource),
        depends_on: depends_on.into_iter().map(InstructionId::new).collect(),
        inputs: inputs.into_iter().map(TensorId::new).collect(),
        outputs: outputs.into_iter().map(TensorId::new).collect(),
        nbytes,
        memory_tier,
        predicted_duration_s,
        executable_ref: executable_ref.map(RegionId::new),
        source: source.map(ResourceId::new),
        destination: destination.map(ResourceId::new),
        backend_id,
        transfer_backend,
        sync_required,
        stream_id: stream_id.map(StreamId::new),
        copy_engine_id,
        link_id,
        io_queue_id,
        attributes,
    })
}

fn attributes_get_str(obj: &Bound<'_, PyAny>, key: &str) -> Option<String> {
    let attrs = obj.getattr("attributes").ok()?;
    if let Ok(dict) = attrs.cast::<PyDict>() {
        let v = dict.get_item(key).ok()??;
        return v.extract().ok();
    }
    if attrs.hasattr("get").ok()? {
        let v = attrs.call_method1("get", (key,)).ok()?;
        if v.is_none() {
            return None;
        }
        return v.extract().ok();
    }
    None
}

fn default_stream_id(opcode: Opcode, resource: &str) -> Option<String> {
    match opcode {
        Opcode::Compute => Some(format!("{resource}::compute")),
        Opcode::Transfer | Opcode::Load | Opcode::Prefetch => Some(format!("{resource}::copy0")),
        Opcode::RecordEvent | Opcode::WaitEvent => Some(format!("{resource}::sync")),
        Opcode::Evict | Opcode::Release => Some(format!("{resource}::lifetime")),
    }
}

pub(crate) fn schedule_from_py(obj: &Bound<'_, PyAny>) -> PyResult<ExecutableSchedule> {
    SCHEDULE_FROM_PY_CALLS.fetch_add(1, Ordering::Relaxed);
    let graph_name: String = obj.getattr("graph_name")?.extract()?;
    let fingerprint: String = obj.getattr("fingerprint")?.extract()?;
    let notes = py_string_seq(&obj.getattr("notes")?)?;
    let inst_list = obj.getattr("instructions")?;
    let mut instructions = Vec::new();
    for item in inst_list.try_iter()? {
        instructions.push(instruction_from_py(&item?)?);
    }
    Ok(ExecutableSchedule::new(
        graph_name,
        fingerprint,
        instructions,
        notes,
    ))
}

fn instruction_to_dict<'py>(py: Python<'py>, inst: &Instruction) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("opcode", inst.opcode.as_str())?;
    d.set_item("name", inst.name.as_str())?;
    d.set_item("resource", inst.resource.as_str())?;
    d.set_item(
        "depends_on",
        inst.depends_on
            .iter()
            .map(|x| x.as_str().to_owned())
            .collect::<Vec<_>>(),
    )?;
    d.set_item(
        "inputs",
        inst.inputs
            .iter()
            .map(|x| x.as_str().to_owned())
            .collect::<Vec<_>>(),
    )?;
    d.set_item(
        "outputs",
        inst.outputs
            .iter()
            .map(|x| x.as_str().to_owned())
            .collect::<Vec<_>>(),
    )?;
    d.set_item("nbytes", inst.nbytes)?;
    d.set_item("memory_tier", inst.memory_tier.as_str())?;
    d.set_item("predicted_duration_s", inst.predicted_duration_s)?;
    d.set_item(
        "executable_ref",
        inst.executable_ref.as_ref().map(|r| r.as_str()),
    )?;
    d.set_item("source", inst.source.as_ref().map(|r| r.as_str()))?;
    d.set_item("destination", inst.destination.as_ref().map(|r| r.as_str()))?;
    d.set_item("backend_id", &inst.backend_id)?;
    d.set_item("transfer_backend", &inst.transfer_backend)?;
    d.set_item("sync_required", inst.sync_required)?;
    d.set_item("stream_id", inst.stream_id.as_ref().map(|s| s.as_str()))?;
    d.set_item("copy_engine_id", &inst.copy_engine_id)?;
    d.set_item("link_id", &inst.link_id)?;
    d.set_item("io_queue_id", &inst.io_queue_id)?;
    let attrs = PyDict::new(py);
    for (k, v) in &inst.attributes {
        attrs.set_item(k, attr_to_py(py, v)?)?;
    }
    d.set_item("attributes", attrs)?;
    Ok(d)
}

pub(crate) fn schedule_to_dict<'py>(
    py: Python<'py>,
    schedule: &ExecutableSchedule,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("graph_name", &schedule.graph_name)?;
    d.set_item("fingerprint", &schedule.fingerprint)?;
    d.set_item("notes", &schedule.notes)?;
    let list = PyList::empty(py);
    for inst in &schedule.instructions {
        list.append(instruction_to_dict(py, inst)?)?;
    }
    d.set_item("instructions", list)?;
    Ok(d)
}

#[pyfunction]
fn schedule_to_json(schedule: &Bound<'_, PyAny>) -> PyResult<String> {
    let s = schedule_from_py(schedule)?;
    s.to_json()
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
fn schedule_from_json(py: Python<'_>, json: &str) -> PyResult<Py<PyAny>> {
    let s =
        ExecutableSchedule::from_json(json).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(schedule_to_dict(py, &s)?.into())
}

#[pyfunction]
fn schedule_roundtrip(py: Python<'_>, schedule: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let s = schedule_from_py(schedule)?;
    let bytes = s
        .to_json_bytes()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let back = ExecutableSchedule::from_json_bytes(&bytes)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(schedule_to_dict(py, &back)?.into())
}

#[pyfunction]
fn validate_schedule_py(schedule: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    let s = schedule_from_py(schedule)?;
    Ok(validate_schedule(&s).errors)
}

#[pyfunction]
fn assert_schedule_valid_py(schedule: &Bound<'_, PyAny>) -> PyResult<()> {
    let s = schedule_from_py(schedule)?;
    assert_schedule_valid(&s).map_err(|e| PyValueError::new_err(e.to_string()))
}

fn machine_from_py(obj: Option<&Bound<'_, PyAny>>) -> PyResult<MachineModel> {
    let Some(obj) = obj else {
        return Ok(MachineModel::cpu_only());
    };
    if obj.is_none() {
        return Ok(MachineModel::cpu_only());
    }
    let mut machine = MachineModel::default();
    if let Ok(compute) = obj.getattr("compute") {
        let items = if compute.hasattr("items")? {
            Some(compute.call_method0("items")?)
        } else if let Ok(dict) = compute.cast::<PyDict>() {
            for (k, v) in dict.iter() {
                let name: String = k.extract()?;
                machine.compute.insert(name.clone(), 1.0);
                if let Ok(aff) = v.getattr("memory_affinity") {
                    if let Ok(seq) = aff.extract::<Vec<String>>() {
                        if let Some(first) = seq.first() {
                            machine.memory_affinity.insert(name, first.clone());
                        }
                    } else if let Ok(mut iter) = aff.try_iter() {
                        if let Some(Ok(first)) = iter.next() {
                            if let Ok(s) = first.extract::<String>() {
                                machine.memory_affinity.insert(name, s);
                            }
                        }
                    }
                }
            }
            None
        } else {
            None
        };
        if let Some(items) = items {
            for item in items.try_iter()? {
                let item = item?;
                let name: String = item.get_item(0)?.extract()?;
                let comp = item.get_item(1)?;
                machine.compute.insert(name.clone(), 1.0);
                if let Ok(aff) = comp.getattr("memory_affinity") {
                    if let Ok(seq) = aff.extract::<Vec<String>>() {
                        if let Some(first) = seq.first() {
                            machine.memory_affinity.insert(name, first.clone());
                        }
                    } else if let Ok(mut iter) = aff.try_iter() {
                        if let Some(Ok(first)) = iter.next() {
                            if let Ok(s) = first.extract::<String>() {
                                machine.memory_affinity.insert(name, s);
                            }
                        }
                    }
                }
            }
        }
    }
    if machine.compute.is_empty() {
        machine.compute.insert("cpu".into(), 1.0);
    }
    if let Ok(memory) = obj.getattr("memory") {
        let items = if memory.hasattr("items")? {
            memory.call_method0("items")?
        } else {
            return Ok(machine);
        };
        for item in items.try_iter()? {
            let item = item?;
            let name: String = item.get_item(0)?.extract()?;
            let mem = item.get_item(1)?;
            let capacity: u64 = mem
                .getattr("capacity_bytes")
                .ok()
                .and_then(|v| v.extract::<i64>().ok())
                .unwrap_or(1 << 30)
                .max(0) as u64;
            let allocatable: u64 = mem
                .getattr("allocatable_bytes")
                .ok()
                .and_then(|v| v.extract::<i64>().ok())
                .unwrap_or(0)
                .max(0) as u64;
            let class: String = mem
                .getattr("memory_class")
                .ok()
                .and_then(|v| {
                    v.extract::<String>()
                        .or_else(|_| v.getattr("value").and_then(|x| x.extract()))
                        .ok()
                })
                .unwrap_or_else(|| "host".into());
            machine.memory.insert(
                name.clone(),
                MemoryResource {
                    name,
                    capacity_bytes: capacity,
                    allocatable_bytes: allocatable,
                    memory_class: class,
                },
            );
        }
    }
    if machine.memory.is_empty() {
        machine.memory.insert(
            "host_ram".into(),
            MemoryResource {
                name: "host_ram".into(),
                capacity_bytes: 64 * 1024 * 1024 * 1024,
                allocatable_bytes: 0,
                memory_class: "host".into(),
            },
        );
    }
    if let Ok(links) = obj.getattr("links") {
        let values = if links.hasattr("values")? {
            links.call_method0("values")?
        } else {
            links
        };
        for item in values.try_iter()? {
            let link = item?;
            if link.is_instance_of::<pyo3::types::PyString>() {
                continue;
            }
            let source: String = match link.getattr("source") {
                Ok(s) => s.extract().unwrap_or_default(),
                Err(_) => continue,
            };
            let destination: String = match link.getattr("destination") {
                Ok(s) => s.extract().unwrap_or_default(),
                Err(_) => continue,
            };
            if source.is_empty() || destination.is_empty() {
                continue;
            }
            let bw: f64 = link
                .getattr("bytes_per_s")
                .ok()
                .or_else(|| link.getattr("bandwidth_bytes_per_s").ok())
                .and_then(|v| if v.is_none() { None } else { v.extract().ok() })
                .unwrap_or(12e9);
            let lat: f64 = link
                .getattr("latency_s")
                .ok()
                .and_then(|v| if v.is_none() { None } else { v.extract().ok() })
                .unwrap_or(1e-5);
            machine.links.push(TransferLink {
                source,
                destination,
                bandwidth_bytes_per_s: bw,
                latency_s: lat,
            });
        }
    }
    Ok(machine)
}

#[pyfunction]
#[pyo3(signature = (schedule, machine=None))]
fn simulate_schedule_py(
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
fn execute_schedule_py(
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
fn execute_schedule_json(py: Python<'_>, json: &str, dry_run: bool) -> PyResult<Py<PyAny>> {
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
    m.add("execute_schedule", m.getattr("execute_schedule_py")?)?;
    Ok(())
}
