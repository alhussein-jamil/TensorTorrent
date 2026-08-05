//! Schedule conversion between Python objects and `ExecutableSchedule`.

use crate::counters::SCHEDULE_FROM_PY_CALLS;
use indexmap::IndexMap;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList};
use std::collections::BTreeMap;
use std::str::FromStr;
use std::sync::atomic::Ordering;
use tt_ir::{
    assert_schedule_valid, validate_schedule, AttrValue, ExecutableSchedule, Instruction,
    InstructionId, MemoryTier, Opcode, RegionId, ResourceId, StreamId, TensorId,
};

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
pub(crate) fn schedule_to_json(schedule: &Bound<'_, PyAny>) -> PyResult<String> {
    let s = schedule_from_py(schedule)?;
    s.to_json()
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
pub(crate) fn schedule_from_json(py: Python<'_>, json: &str) -> PyResult<Py<PyAny>> {
    let s =
        ExecutableSchedule::from_json(json).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(schedule_to_dict(py, &s)?.into())
}

#[pyfunction]
pub(crate) fn schedule_roundtrip(
    py: Python<'_>,
    schedule: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let s = schedule_from_py(schedule)?;
    let bytes = s
        .to_json_bytes()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let back = ExecutableSchedule::from_json_bytes(&bytes)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(schedule_to_dict(py, &back)?.into())
}

#[pyfunction]
pub(crate) fn validate_schedule_py(schedule: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    let s = schedule_from_py(schedule)?;
    Ok(validate_schedule(&s).errors)
}

#[pyfunction]
pub(crate) fn assert_schedule_valid_py(schedule: &Bound<'_, PyAny>) -> PyResult<()> {
    let s = schedule_from_py(schedule)?;
    assert_schedule_valid(&s).map_err(|e| PyValueError::new_err(e.to_string()))
}
