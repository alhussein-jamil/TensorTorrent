//! Machine model conversion from Python.

use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use tt_runtime::{MachineModel, MemoryResource, TransferLink};

pub(crate) fn machine_from_py(obj: Option<&Bound<'_, PyAny>>) -> PyResult<MachineModel> {
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
