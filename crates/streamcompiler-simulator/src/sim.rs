//! Discrete-event walk of an ExecutableSchedule DAG.

use crate::machine::MachineModel;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use streamcompiler_core::{assert_schedule_valid, ExecutableSchedule, Opcode};
use streamcompiler_core::{AllocationId, ResourceId, TensorId};
use streamcompiler_memory::{AllocationTable, ResidencyStore, TensorMetadata};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TimelineEvent {
    pub name: String,
    pub opcode: String,
    pub resource: String,
    pub start_s: f64,
    pub end_s: f64,
    pub nbytes: u64,
    pub simulated: bool,
    pub critical_pred: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SimulationResult {
    pub makespan_s: f64,
    pub peak_bytes: HashMap<String, u64>,
    pub timeline: Vec<TimelineEvent>,
    pub exposed_transfer_latency_s: f64,
    pub resource_busy_s: HashMap<String, f64>,
    pub simulated: bool,
    pub critical_path: Vec<String>,
    pub bytes_read: u64,
    pub bytes_transferred: u64,
    pub instruction_count: usize,
    pub activation_peak_bytes: u64,
}

/// Simulate `schedule` against `machine`. Rejects invalid/incomplete schedules.
pub fn simulate_schedule(
    schedule: &ExecutableSchedule,
    machine: &MachineModel,
) -> Result<SimulationResult, streamcompiler_core::CoreError> {
    assert_schedule_valid(schedule)?;

    let by_name: HashMap<&str, &streamcompiler_core::Instruction> = schedule
        .instructions
        .iter()
        .map(|i| (i.name.as_str(), i))
        .collect();
    let mut remaining: HashMap<&str, HashSet<&str>> = schedule
        .instructions
        .iter()
        .map(|i| {
            (
                i.name.as_str(),
                i.depends_on.iter().map(|d| d.as_str()).collect(),
            )
        })
        .collect();
    let mut dependents: HashMap<&str, Vec<&str>> = HashMap::new();
    for inst in &schedule.instructions {
        for dep in &inst.depends_on {
            dependents
                .entry(dep.as_str())
                .or_default()
                .push(inst.name.as_str());
        }
    }

    let mut ready: VecDeque<&str> = remaining
        .iter()
        .filter_map(|(n, deps)| if deps.is_empty() { Some(*n) } else { None })
        .collect();

    let mut compute_free: HashMap<String, f64> =
        machine.compute.keys().map(|k| (k.clone(), 0.0)).collect();
    let mut copy_free: HashMap<String, f64> = HashMap::new();
    let mut io_free = 0.0f64;
    let mut resource_busy: HashMap<String, f64> =
        machine.compute.keys().map(|k| (k.clone(), 0.0)).collect();
    let mut peak: HashMap<String, u64> = machine.memory.keys().map(|k| (k.clone(), 0u64)).collect();
    let mut resident: HashMap<String, u64> =
        machine.memory.keys().map(|k| (k.clone(), 0u64)).collect();

    let allocations = Arc::new(AllocationTable::new());
    for (name, mem) in &machine.memory {
        allocations.set_capacity_limit(name, mem.capacity_bytes);
    }
    let residency = ResidencyStore::new(Arc::clone(&allocations));

    let mut event_ready_at: HashMap<String, f64> = HashMap::new();
    let mut inst_end: HashMap<String, f64> = HashMap::new();
    let mut timeline = Vec::new();
    let mut bytes_read = 0u64;
    let mut bytes_transferred = 0u64;
    let mut exposed = 0.0f64;
    let mut cp_pred: HashMap<String, Option<String>> = HashMap::new();
    let mut cp_finish: HashMap<String, f64> = HashMap::new();
    let mut last_on_compute: HashMap<String, String> = HashMap::new();
    let mut last_on_copy: HashMap<String, String> = HashMap::new();
    let mut last_on_io: Option<String> = None;
    let mut activation_peak = 0u64;
    let mut alloc_counter = 0u64;

    while let Some(name) = ready.pop_front() {
        let inst = by_name[name];
        let dep_end = inst
            .depends_on
            .iter()
            .map(|d| inst_end.get(d.as_str()).copied().unwrap_or(0.0))
            .fold(0.0f64, f64::max);

        let mut pred = inst
            .depends_on
            .iter()
            .max_by(|a, b| {
                let ea = inst_end.get(a.as_str()).copied().unwrap_or(0.0);
                let eb = inst_end.get(b.as_str()).copied().unwrap_or(0.0);
                ea.partial_cmp(&eb).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|d| d.as_str().to_owned());

        let (start, end, nbytes) = match inst.opcode {
            Opcode::Compute => {
                let res = inst.resource.as_str();
                let free = compute_free.get(res).copied().unwrap_or(0.0);
                if let Some(prev) = last_on_compute.get(res) {
                    if free >= dep_end {
                        pred = Some(prev.clone());
                    }
                }
                let start = dep_end.max(free);
                let mut dur = inst.predicted_duration_s;
                if let Some(d) = inst.attributes.get("mock_compute_delay_s") {
                    if let Some(v) = d.as_i64().map(|i| i as f64).or(match d {
                        streamcompiler_core::AttrValue::Float(f) => Some(*f),
                        _ => None,
                    }) {
                        dur = dur.max(v);
                    }
                }
                if dur <= 0.0 {
                    dur = 1e-6;
                }
                let end = start + dur;
                compute_free.insert(res.to_owned(), end);
                *resource_busy.entry(res.to_owned()).or_insert(0.0) += dur;
                last_on_compute.insert(res.to_owned(), name.to_owned());

                // Track output allocations.
                for out in &inst.outputs {
                    let n = tensor_nbytes(inst, out.as_str());
                    alloc_counter += 1;
                    let aid = AllocationId::new(format!("sim-{alloc_counter}"));
                    let meta = TensorMetadata {
                        nbytes: n,
                        dtype: "unknown".into(),
                        ..Default::default()
                    };
                    let _ = residency.put(
                        TensorId::new(out.as_str()),
                        ResourceId::new(res),
                        aid,
                        meta,
                        None,
                    );
                    bump_mem(&mut resident, &mut peak, mem_for(res, machine), n);
                }
                activation_peak = activation_peak.max(allocations.live_bytes());
                (start, end, inst.nbytes)
            }
            Opcode::Transfer => {
                let src = inst.source.as_ref().map(|s| s.as_str()).unwrap_or("");
                let dst = inst
                    .destination
                    .as_ref()
                    .map(|s| s.as_str())
                    .unwrap_or(inst.resource.as_str());
                let link_key = format!("{src}->{dst}");
                let free = copy_free.get(&link_key).copied().unwrap_or(0.0);
                if let Some(prev) = last_on_copy.get(&link_key) {
                    if free >= dep_end {
                        pred = Some(prev.clone());
                    }
                }
                let start = dep_end.max(free);
                let n = inst.nbytes.max(1);
                let mut dur = machine.transfer_time(src, dst, n);
                if let Some(d) = inst.attributes.get("mock_transfer_delay_s") {
                    if let Some(v) = match d {
                        streamcompiler_core::AttrValue::Float(f) => Some(*f),
                        streamcompiler_core::AttrValue::Int(i) => Some(*i as f64),
                        _ => None,
                    } {
                        dur = dur.max(v);
                    }
                }
                let end = start + dur;
                copy_free.insert(link_key.clone(), end);
                last_on_copy.insert(link_key, name.to_owned());
                bytes_transferred += n;
                exposed += dur;

                for tid in inst.outputs.iter().chain(inst.inputs.iter()) {
                    alloc_counter += 1;
                    let aid = AllocationId::new(format!("sim-{alloc_counter}"));
                    let _ = residency.replicate(
                        &TensorId::new(tid.as_str()),
                        ResourceId::new(dst),
                        aid,
                        None,
                    );
                    // If tensor wasn't on src, put first.
                    if residency
                        .get(&TensorId::new(tid.as_str()), &ResourceId::new(src))
                        .is_err()
                    {
                        let meta = TensorMetadata {
                            nbytes: tensor_nbytes(inst, tid.as_str()),
                            ..Default::default()
                        };
                        let _ = residency.put(
                            TensorId::new(tid.as_str()),
                            ResourceId::new(src),
                            AllocationId::new(format!("sim-src-{alloc_counter}")),
                            meta,
                            None,
                        );
                        let _ = residency.replicate(
                            &TensorId::new(tid.as_str()),
                            ResourceId::new(dst),
                            AllocationId::new(format!("sim-dst-{alloc_counter}")),
                            None,
                        );
                    }
                    bump_mem(
                        &mut resident,
                        &mut peak,
                        mem_for(dst, machine),
                        tensor_nbytes(inst, tid.as_str()),
                    );
                }
                activation_peak = activation_peak.max(allocations.live_bytes());
                (start, end, n)
            }
            Opcode::Prefetch | Opcode::Load => {
                if let Some(prev) = &last_on_io {
                    if io_free >= dep_end {
                        pred = Some(prev.clone());
                    }
                }
                let start = dep_end.max(io_free);
                let n = inst.nbytes.max(1);
                let dur = machine.transfer_time("disk", inst.resource.as_str(), n);
                let end = start + dur;
                io_free = end;
                last_on_io = Some(name.to_owned());
                bytes_read += n;
                let dest = inst
                    .destination
                    .as_ref()
                    .map(|d| d.as_str())
                    .unwrap_or(inst.resource.as_str());
                for tid in inst.outputs.iter().chain(inst.inputs.iter()) {
                    alloc_counter += 1;
                    let meta = TensorMetadata {
                        nbytes: tensor_nbytes(inst, tid.as_str()),
                        ..Default::default()
                    };
                    let _ = residency.put(
                        TensorId::new(tid.as_str()),
                        ResourceId::new(dest),
                        AllocationId::new(format!("sim-{alloc_counter}")),
                        meta,
                        None,
                    );
                    bump_mem(
                        &mut resident,
                        &mut peak,
                        mem_for(dest, machine),
                        tensor_nbytes(inst, tid.as_str()),
                    );
                }
                activation_peak = activation_peak.max(allocations.live_bytes());
                (start, end, n)
            }
            Opcode::RecordEvent => {
                let start = dep_end;
                let end = start;
                event_ready_at.insert(name.to_owned(), end);
                (start, end, 0)
            }
            Opcode::WaitEvent => {
                let waits = inst
                    .attr_str("waits_for")
                    .map(str::to_owned)
                    .or_else(|| inst.depends_on.first().map(|d| d.as_str().to_owned()));
                let event_t = waits
                    .as_ref()
                    .and_then(|w| event_ready_at.get(w).copied())
                    .unwrap_or(dep_end);
                let start = dep_end.max(event_t);
                (start, start, 0)
            }
            Opcode::Evict | Opcode::Release => {
                let start = dep_end;
                let end = start;
                let res = inst
                    .attr_str("release_resource")
                    .unwrap_or(inst.resource.as_str());
                for tid in &inst.inputs {
                    let _ =
                        residency.release_copy(&TensorId::new(tid.as_str()), &ResourceId::new(res));
                    let n = tensor_nbytes(inst, tid.as_str());
                    let mem = mem_for(res, machine);
                    if let Some(v) = resident.get_mut(mem) {
                        *v = v.saturating_sub(n);
                    }
                }
                (start, end, inst.nbytes)
            }
        };

        inst_end.insert(name.to_owned(), end);
        cp_finish.insert(name.to_owned(), end);
        cp_pred.insert(name.to_owned(), pred.clone());
        timeline.push(TimelineEvent {
            name: name.to_owned(),
            opcode: inst.opcode.to_string(),
            resource: inst.resource.to_string(),
            start_s: start,
            end_s: end,
            nbytes,
            simulated: true,
            critical_pred: pred,
        });

        if let Some(nexts) = dependents.get(name) {
            for nxt in nexts {
                if let Some(deps) = remaining.get_mut(nxt) {
                    deps.remove(name);
                    if deps.is_empty() {
                        ready.push_back(nxt);
                    }
                }
            }
        }
    }

    let makespan = inst_end.values().copied().fold(0.0f64, f64::max);
    let critical_path = reconstruct_critical_path(&cp_pred, &cp_finish, makespan);

    Ok(SimulationResult {
        makespan_s: makespan,
        peak_bytes: peak,
        timeline,
        exposed_transfer_latency_s: exposed,
        resource_busy_s: resource_busy,
        simulated: true,
        critical_path,
        bytes_read,
        bytes_transferred,
        instruction_count: schedule.instructions.len(),
        activation_peak_bytes: activation_peak,
    })
}

fn tensor_nbytes(inst: &streamcompiler_core::Instruction, tensor: &str) -> u64 {
    let sizes = inst.tensor_nbytes();
    if let Some(n) = sizes.get(tensor) {
        return (*n).max(1);
    }
    inst.nbytes.max(1)
}

fn mem_for<'a>(resource: &'a str, machine: &'a MachineModel) -> &'a str {
    if machine.memory.contains_key(resource) {
        return resource;
    }
    let lower = resource.to_lowercase();
    if lower.contains("mock") || lower.contains("cuda") || lower.contains("gpu") {
        for name in machine.memory.keys() {
            if name.contains(resource) || name.contains("vram") {
                return name.as_str();
            }
        }
    }
    machine
        .memory
        .keys()
        .find(|k| k.contains("ram") || k.contains("host"))
        .map(|s| s.as_str())
        .unwrap_or("host_ram")
}

fn bump_mem(
    resident: &mut HashMap<String, u64>,
    peak: &mut HashMap<String, u64>,
    mem: &str,
    nbytes: u64,
) {
    let live = resident.entry(mem.to_owned()).or_insert(0);
    *live = live.saturating_add(nbytes);
    let p = peak.entry(mem.to_owned()).or_insert(0);
    *p = (*p).max(*live);
}

fn reconstruct_critical_path(
    pred: &HashMap<String, Option<String>>,
    finish: &HashMap<String, f64>,
    makespan: f64,
) -> Vec<String> {
    let Some((end_name, _)) = finish
        .iter()
        .filter(|(_, t)| (**t - makespan).abs() < 1e-12)
        .max_by(|a, b| a.0.cmp(b.0))
    else {
        return vec![];
    };
    let mut path = vec![end_name.clone()];
    let mut cur = end_name.clone();
    while let Some(Some(p)) = pred.get(&cur) {
        path.push(p.clone());
        cur = p.clone();
    }
    path.reverse();
    path
}

#[cfg(test)]
mod tests {
    use super::*;
    use indexmap::IndexMap;
    use streamcompiler_core::{
        Instruction, InstructionId, MemoryTier, RegionId, ResourceId, TensorId,
    };

    fn simple_schedule() -> ExecutableSchedule {
        let a = Instruction {
            opcode: Opcode::Compute,
            name: InstructionId::new("compute::a"),
            resource: ResourceId::new("cpu"),
            depends_on: vec![],
            inputs: vec![TensorId::new("in")],
            outputs: vec![TensorId::new("out")],
            nbytes: 64,
            memory_tier: MemoryTier::SystemRam,
            predicted_duration_s: 0.1,
            executable_ref: Some(RegionId::new("a")),
            source: None,
            destination: None,
            backend_id: Some("cpu".into()),
            transfer_backend: None,
            sync_required: false,
            attributes: IndexMap::new(),
        };
        let b = Instruction {
            opcode: Opcode::Compute,
            name: InstructionId::new("compute::b"),
            resource: ResourceId::new("cpu"),
            depends_on: vec![InstructionId::new("compute::a")],
            inputs: vec![TensorId::new("out")],
            outputs: vec![TensorId::new("out2")],
            nbytes: 64,
            memory_tier: MemoryTier::SystemRam,
            predicted_duration_s: 0.2,
            executable_ref: Some(RegionId::new("b")),
            source: None,
            destination: None,
            backend_id: Some("cpu".into()),
            transfer_backend: None,
            sync_required: false,
            attributes: IndexMap::new(),
        };
        ExecutableSchedule::new("g", "fp", vec![a, b], vec![])
    }

    #[test]
    fn deterministic_makespan() {
        let machine = MachineModel::cpu_only();
        let schedule = simple_schedule();
        let r1 = simulate_schedule(&schedule, &machine).unwrap();
        let r2 = simulate_schedule(&schedule, &machine).unwrap();
        assert!((r1.makespan_s - 0.3).abs() < 1e-9);
        assert_eq!(r1.makespan_s, r2.makespan_s);
        assert!(r1.simulated);
        assert_eq!(r1.critical_path, r2.critical_path);
    }
}
