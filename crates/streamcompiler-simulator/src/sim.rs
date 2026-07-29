//! Discrete-event walk of an ExecutableSchedule DAG (parity with Python oracle).

use crate::machine::MachineModel;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value as JsonValue};
use std::collections::{HashMap, HashSet, VecDeque};
use streamcompiler_core::{assert_schedule_valid, ExecutableSchedule, Opcode};

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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub event: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resident_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub allocatable_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub at_s: Option<f64>,
    /// Spill write volume (not RAM residency).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub activation_bytes_written: Option<u64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SimulationResult {
    pub makespan_s: f64,
    pub peak_bytes: HashMap<String, u64>,
    pub timeline: Vec<TimelineEvent>,
    pub transfer_events: Vec<JsonValue>,
    pub release_events: Vec<JsonValue>,
    pub exposed_transfer_latency_s: f64,
    pub resource_busy_s: HashMap<String, f64>,
    pub simulated: bool,
    pub critical_path: Vec<String>,
    pub bytes_read: u64,
    pub bytes_transferred: u64,
    pub instruction_count: usize,
    pub activation_peak_bytes: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InfeasibilityReport {
    pub memory: String,
    pub resident_bytes: u64,
    pub allocatable_bytes: u64,
    pub instruction: String,
    pub at_s: f64,
    pub peak_bytes: HashMap<String, u64>,
    pub simulated: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "status", content = "body")]
pub enum SimulationOutcome {
    Valid(SimulationResult),
    InfeasibleMemory(InfeasibilityReport),
    InvalidResidency { detail: String },
    InvalidEvent { detail: String },
    Unsupported { detail: String },
}

/// Simulate `schedule` against `machine`. Rejects invalid/incomplete schedules.
///
/// Memory overflow is **not** a valid report — it returns
/// [`SimulationOutcome::InfeasibleMemory`].
pub fn simulate_schedule(
    schedule: &ExecutableSchedule,
    machine: &MachineModel,
) -> Result<SimulationOutcome, streamcompiler_core::CoreError> {
    let result = simulate_schedule_inner(schedule, machine)?;
    if let Some(pressure) = result
        .timeline
        .iter()
        .find(|e| e.event.as_deref() == Some("eviction_pressure"))
    {
        return Ok(SimulationOutcome::InfeasibleMemory(InfeasibilityReport {
            memory: pressure.memory.clone().unwrap_or_default(),
            resident_bytes: pressure.resident_bytes.unwrap_or(0),
            allocatable_bytes: pressure.allocatable_bytes.unwrap_or(0),
            instruction: pressure.name.clone(),
            at_s: pressure.at_s.unwrap_or(0.0),
            peak_bytes: result.peak_bytes.clone(),
            simulated: true,
        }));
    }
    Ok(SimulationOutcome::Valid(result))
}

fn simulate_schedule_inner(
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

    // (tensor, resource) -> alloc_id ; alloc_id -> (mem, capacity, refs)
    let mut copies: HashMap<(String, String), String> = HashMap::new();
    let mut allocations: HashMap<String, (String, u64, u32)> = HashMap::new();
    let mut state_leases: Vec<(f64, String, u64)> = Vec::new();

    let mut event_ready_at: HashMap<String, f64> = HashMap::new();
    let mut inst_end: HashMap<String, f64> = HashMap::new();
    let mut timeline = Vec::new();
    let mut transfer_events = Vec::new();
    let mut release_events = Vec::new();
    let mut bytes_read = 0u64;
    let mut bytes_transferred = 0u64;
    let mut exposed = 0.0f64;
    let mut cp_pred: HashMap<String, Option<String>> = HashMap::new();
    let mut cp_finish: HashMap<String, f64> = HashMap::new();
    let mut last_on_compute: HashMap<String, String> = HashMap::new();
    let mut last_on_copy: HashMap<String, String> = HashMap::new();
    let mut last_on_io: Option<String> = None;
    let mut activation_peak = 0u64;
    let mut activation_ids: HashSet<String> = HashSet::new();

    // External inputs (user tensors never produced by Load/Compute/Transfer)
    // start resident on the first Transfer source that consumes them. This is
    // not inventing a missing mid-schedule copy — it models host-provided inputs.
    let produced = produced_tensor_ids(schedule);
    for inst in &schedule.instructions {
        if inst.opcode != Opcode::Transfer {
            continue;
        }
        let src = inst.source.as_ref().map(|s| s.as_str()).unwrap_or("");
        if src.is_empty() {
            continue;
        }
        for tid in unique_tensors(inst) {
            if produced.contains(&tid) {
                continue;
            }
            let key = (tid.clone(), src.to_owned());
            if copies.contains_key(&key) {
                continue;
            }
            let tn = tensor_nbytes(inst, &tid);
            install_copy(
                &mut copies,
                &mut allocations,
                &mut resident,
                &mut peak,
                &mut timeline,
                machine,
                &tid,
                src,
                tn,
                inst,
                0.0,
            );
        }
    }

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

        release_state_due(&mut state_leases, &mut resident, dep_end);

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
                    if let Some(v) = attr_f64(d) {
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

                // Optional state_bytes lease for the kernel window.
                if let Some(state) = inst.attributes.get("state_bytes") {
                    if let Some(n) = attr_u64(state).filter(|n| *n > 0) {
                        let mem = mem_for(res, machine).to_owned();
                        // Only bump if no state tensor already resident from Load.
                        let already = inst
                            .inputs
                            .iter()
                            .any(|t| copies.contains_key(&(t.as_str().to_owned(), res.to_owned())));
                        if !already {
                            bump_mem(
                                &mut resident,
                                &mut peak,
                                &mut timeline,
                                machine,
                                &mem,
                                n,
                                start,
                                name,
                            );
                            state_leases.push((end, mem, n));
                        }
                    }
                }

                for out in &inst.outputs {
                    let out = out.as_str().to_owned();
                    let n = tensor_nbytes(inst, &out);
                    install_copy(
                        &mut copies,
                        &mut allocations,
                        &mut resident,
                        &mut peak,
                        &mut timeline,
                        machine,
                        &out,
                        res,
                        n,
                        inst,
                        start,
                    );
                    activation_ids.insert(out);
                }
                activation_peak = activation_peak.max(live_activation_bytes(
                    &copies,
                    &allocations,
                    &activation_ids,
                ));
                (start, end, inst.nbytes)
            }
            Opcode::Transfer => {
                let src = inst.source.as_ref().map(|s| s.as_str()).unwrap_or("");
                let dst = inst
                    .destination
                    .as_ref()
                    .map(|s| s.as_str())
                    .unwrap_or(inst.resource.as_str());
                let engine = inst.resource.as_str();
                let free = copy_free.get(engine).copied().unwrap_or(0.0);
                if let Some(prev) = last_on_copy.get(engine) {
                    if free >= dep_end {
                        pred = Some(prev.clone());
                    }
                }
                let start = dep_end.max(free);
                let n = inst.nbytes.max(1);
                let mut dur = machine.transfer_time(src, dst, n);
                if let Some(d) = inst.attributes.get("mock_transfer_delay_s") {
                    if let Some(v) = attr_f64(d) {
                        dur = dur.max(v);
                    }
                }
                let end = start + dur;
                copy_free.insert(engine.to_owned(), end);
                last_on_copy.insert(engine.to_owned(), name.to_owned());
                bytes_transferred += n;
                exposed += dur;

                for tid in unique_tensors(inst) {
                    let tn = tensor_nbytes(inst, &tid);
                    // Strict: never invent a missing source copy.
                    if !copies.contains_key(&(tid.clone(), src.to_owned())) {
                        return Err(streamcompiler_core::CoreError::Validation(format!(
                            "transfer {:?} source copy missing for tensor {} on {}",
                            inst.name.as_str(),
                            tid,
                            src
                        )));
                    }
                    install_copy(
                        &mut copies,
                        &mut allocations,
                        &mut resident,
                        &mut peak,
                        &mut timeline,
                        machine,
                        &tid,
                        dst,
                        tn,
                        inst,
                        start,
                    );
                }
                transfer_events.push(json!({
                    "event": "Transfer",
                    "instruction": name,
                    "source": src,
                    "destination": dst,
                    "nbytes": n,
                    "start_s": start,
                    "end_s": end,
                    "contention_factor": 1.0,
                    "simulated": true,
                }));
                activation_peak = activation_peak.max(live_activation_bytes(
                    &copies,
                    &allocations,
                    &activation_ids,
                ));
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
                let dest = inst
                    .destination
                    .as_ref()
                    .map(|d| d.as_str())
                    .unwrap_or(inst.resource.as_str());
                let dur = machine.transfer_time("disk", dest, n);
                let end = start + dur;
                io_free = end;
                last_on_io = Some(name.to_owned());
                bytes_read += n;
                let kind = inst.attr_str("kind").unwrap_or("");
                for tid in unique_tensors(inst) {
                    let tn = tensor_nbytes(inst, &tid);
                    install_copy(
                        &mut copies,
                        &mut allocations,
                        &mut resident,
                        &mut peak,
                        &mut timeline,
                        machine,
                        &tid,
                        dest,
                        tn,
                        inst,
                        start,
                    );
                    if kind == "activation_reload" {
                        activation_ids.insert(tid);
                    }
                }
                if kind == "activation_reload" {
                    activation_peak = activation_peak.max(live_activation_bytes(
                        &copies,
                        &allocations,
                        &activation_ids,
                    ));
                }
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
                let Some(wf) = waits else {
                    return Err(streamcompiler_core::CoreError::Validation(format!(
                        "WaitEvent {:?} missing waits_for / depends_on",
                        inst.name.as_str()
                    )));
                };
                // Strict: never invent a missing RecordEvent timestamp.
                let Some(&event_t) = event_ready_at.get(&wf) else {
                    return Err(streamcompiler_core::CoreError::Validation(format!(
                        "WaitEvent {:?} references unknown event {wf:?}",
                        inst.name.as_str()
                    )));
                };
                let start = dep_end.max(event_t);
                (start, start, 0)
            }
            Opcode::Evict | Opcode::Release => {
                let kind = inst.attr_str("kind").unwrap_or("");
                let start = dep_end;
                if inst.opcode == Opcode::Evict && kind == "activation_spill" {
                    if let Some(prev) = &last_on_io {
                        if io_free >= dep_end {
                            pred = Some(prev.clone());
                        }
                    }
                    let n = inst.nbytes.max(1);
                    let start = dep_end.max(io_free);
                    let dur = (n as f64) / (500.0 * (1 << 20) as f64);
                    let end = start + dur.max(1e-6);
                    io_free = end;
                    last_on_io = Some(name.to_owned());
                    let res = inst
                        .attr_str("spill_resource")
                        .or_else(|| inst.source.as_ref().map(|s| s.as_str()))
                        .unwrap_or(inst.resource.as_str());
                    let mut freed_total = 0u64;
                    let mut written = 0u64;
                    for tid in &inst.inputs {
                        let tn = tensor_nbytes(inst, tid.as_str()).max(1);
                        freed_total += drop_copy(
                            &mut copies,
                            &mut allocations,
                            &mut resident,
                            tid.as_str(),
                            res,
                        );
                        written += tn;
                        install_copy(
                            &mut copies,
                            &mut allocations,
                            &mut resident,
                            &mut peak,
                            &mut timeline,
                            machine,
                            tid.as_str(),
                            "disk",
                            tn,
                            inst,
                            end,
                        );
                    }
                    activation_peak = activation_peak.max(live_activation_bytes(
                        &copies,
                        &allocations,
                        &activation_ids,
                    ));
                    release_events.push(json!({
                        "event": "Evict",
                        "instruction": name,
                        "nbytes": freed_total,
                        "activation_bytes_written": written,
                        "start_s": start,
                        "end_s": end,
                        "simulated": true,
                        "notes": "activation_spill RAM→disk",
                    }));
                    inst_end.insert(name.to_owned(), end);
                    cp_finish.insert(name.to_owned(), end);
                    cp_pred.insert(name.to_owned(), pred.clone());
                    timeline.push(TimelineEvent {
                        name: name.to_owned(),
                        opcode: inst.opcode.to_string(),
                        resource: res.to_owned(),
                        start_s: start,
                        end_s: end,
                        nbytes: freed_total,
                        simulated: true,
                        critical_pred: pred,
                        event: Some("Evict".into()),
                        memory: None,
                        resident_bytes: None,
                        allocatable_bytes: None,
                        at_s: Some(start),
                        activation_bytes_written: Some(written),
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
                    continue;
                }
                let end = start;
                let res = inst
                    .attr_str("release_resource")
                    .unwrap_or(inst.resource.as_str());
                let mut freed_total = 0u64;
                for tid in &inst.inputs {
                    let freed = drop_copy(
                        &mut copies,
                        &mut allocations,
                        &mut resident,
                        tid.as_str(),
                        res,
                    );
                    freed_total += freed;
                    release_events.push(json!({
                        "event": "Release",
                        "instruction": name,
                        "tensor": tid.as_str(),
                        "resource": res,
                        "nbytes": freed,
                        "at_s": start,
                        "simulated": true,
                    }));
                }
                activation_peak = activation_peak.max(live_activation_bytes(
                    &copies,
                    &allocations,
                    &activation_ids,
                ));
                (start, end, freed_total.max(inst.nbytes))
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
            event: None,
            memory: None,
            resident_bytes: None,
            allocatable_bytes: None,
            at_s: None,
            activation_bytes_written: None,
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
        transfer_events,
        release_events,
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

fn produced_tensor_ids(schedule: &ExecutableSchedule) -> HashSet<String> {
    let mut out = HashSet::new();
    for inst in &schedule.instructions {
        match inst.opcode {
            Opcode::Load | Opcode::Prefetch => {
                for t in inst.outputs.iter().chain(inst.inputs.iter()) {
                    out.insert(t.as_str().to_owned());
                }
            }
            Opcode::Compute => {
                for t in &inst.outputs {
                    out.insert(t.as_str().to_owned());
                }
            }
            // Transfer replicates an existing copy — it does not create the source.
            _ => {}
        }
    }
    out
}

fn unique_tensors(inst: &streamcompiler_core::Instruction) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for t in inst.outputs.iter().chain(inst.inputs.iter()) {
        let s = t.as_str().to_owned();
        if seen.insert(s.clone()) {
            out.push(s);
        }
    }
    if out.is_empty() && inst.nbytes > 0 {
        out.push(format!("anon::{}", inst.name.as_str()));
    }
    out
}

fn tensor_nbytes(inst: &streamcompiler_core::Instruction, tensor: &str) -> u64 {
    let sizes = inst.tensor_nbytes();
    if let Some(n) = sizes.get(tensor) {
        return *n; // may be 0 — match Python (no fake .max(1) for residency)
    }
    for key in ["input_bytes", "output_bytes"] {
        if let Some(streamcompiler_core::AttrValue::IntMap(m)) = inst.attributes.get(key) {
            if let Some(n) = m.get(tensor) {
                return (*n).max(0) as u64;
            }
        }
    }
    if inst.outputs.len() == 1 && inst.outputs[0].as_str() == tensor {
        return inst.nbytes;
    }
    if inst.inputs.len() == 1 && inst.inputs[0].as_str() == tensor {
        return inst.nbytes;
    }
    inst.nbytes
}

fn allocation_id(inst: &streamcompiler_core::Instruction, tensor: &str, resource: &str) -> String {
    if let Some(streamcompiler_core::AttrValue::StringMap(m)) =
        inst.attributes.get("allocation_ids")
    {
        if let Some(id) = m.get(tensor) {
            return id.clone();
        }
    }
    if let Some(streamcompiler_core::AttrValue::Map(m)) = inst.attributes.get("allocation_ids") {
        if let Some(streamcompiler_core::AttrValue::String(id)) = m.get(tensor) {
            return id.clone();
        }
    }
    format!("sim::{resource}::{tensor}")
}

fn mem_for<'a>(resource: &'a str, machine: &'a MachineModel) -> &'a str {
    if machine.memory.contains_key(resource) {
        return resource;
    }
    if let Some(aff) = machine.memory_affinity.get(resource) {
        if machine.memory.contains_key(aff) {
            return aff.as_str();
        }
    }
    let lower = resource.to_lowercase();
    let hostish = ["cpu", "host", "numa", "pinned", "system_ram", "disk"]
        .iter()
        .any(|t| lower == *t || lower.contains(t));
    if hostish {
        for (name, mem) in &machine.memory {
            let cls = mem.memory_class.to_lowercase();
            let n = name.to_lowercase();
            if n.contains("vram") || cls.contains("device") {
                continue;
            }
            if n.contains("ram") || n.contains("host") || n.contains("numa") {
                return name.as_str();
            }
        }
        return "host_ram";
    }
    // Device compute without affinity: match vram_* by trailing digit.
    let digit: String = lower.chars().filter(|c| c.is_ascii_digit()).collect();
    if !digit.is_empty() {
        for (name, mem) in &machine.memory {
            let cls = mem.memory_class.to_lowercase();
            let n = name.to_lowercase();
            if (n.contains("vram") || cls.contains("device")) && n.contains(&digit) {
                return name.as_str();
            }
        }
    }
    for (name, mem) in &machine.memory {
        let cls = mem.memory_class.to_lowercase();
        if name.to_lowercase().contains("vram") || cls.contains("device") {
            return name.as_str();
        }
    }
    machine
        .memory
        .keys()
        .next()
        .map(|s| s.as_str())
        .unwrap_or("host_ram")
}

#[allow(clippy::too_many_arguments)]
fn bump_mem(
    resident: &mut HashMap<String, u64>,
    peak: &mut HashMap<String, u64>,
    timeline: &mut Vec<TimelineEvent>,
    machine: &MachineModel,
    mem: &str,
    nbytes: u64,
    at_s: f64,
    reason: &str,
) {
    if nbytes == 0 {
        return;
    }
    let live = resident.entry(mem.to_owned()).or_insert(0);
    *live = live.saturating_add(nbytes);
    let p = peak.entry(mem.to_owned()).or_insert(0);
    *p = (*p).max(*live);
    let allocatable = machine
        .memory
        .get(mem)
        .map(|m| {
            if m.allocatable_bytes > 0 {
                m.allocatable_bytes
            } else {
                m.capacity_bytes
            }
        })
        .unwrap_or(0);
    if allocatable > 0 && *live > allocatable {
        timeline.push(TimelineEvent {
            name: reason.to_owned(),
            opcode: "EvictionPressure".into(),
            resource: mem.to_owned(),
            start_s: at_s,
            end_s: at_s,
            nbytes: 0,
            simulated: true,
            critical_pred: None,
            event: Some("eviction_pressure".into()),
            memory: Some(mem.to_owned()),
            resident_bytes: Some(*live),
            allocatable_bytes: Some(allocatable),
            at_s: Some(at_s),
            activation_bytes_written: None,
        });
    }
}

#[allow(clippy::too_many_arguments)]
fn install_copy(
    copies: &mut HashMap<(String, String), String>,
    allocations: &mut HashMap<String, (String, u64, u32)>,
    resident: &mut HashMap<String, u64>,
    peak: &mut HashMap<String, u64>,
    timeline: &mut Vec<TimelineEvent>,
    machine: &MachineModel,
    tensor: &str,
    resource: &str,
    nbytes: u64,
    inst: &streamcompiler_core::Instruction,
    at_s: f64,
) {
    let key = (tensor.to_owned(), resource.to_owned());
    let new_alloc = allocation_id(inst, tensor, resource);
    if copies.get(&key).map(|a| a == &new_alloc).unwrap_or(false) {
        return;
    }
    if copies.contains_key(&key) {
        let _ = drop_copy(copies, allocations, resident, tensor, resource);
    }
    let mem = mem_for(resource, machine).to_owned();
    match allocations.get_mut(&new_alloc) {
        Some(rec) => {
            rec.1 = rec.1.max(nbytes);
            rec.2 = rec.2.saturating_add(1);
        }
        None => {
            allocations.insert(new_alloc.clone(), (mem.clone(), nbytes, 1));
            bump_mem(
                resident,
                peak,
                timeline,
                machine,
                &mem,
                nbytes,
                at_s,
                inst.name.as_str(),
            );
        }
    }
    copies.insert(key, new_alloc);
}

fn drop_copy(
    copies: &mut HashMap<(String, String), String>,
    allocations: &mut HashMap<String, (String, u64, u32)>,
    resident: &mut HashMap<String, u64>,
    tensor: &str,
    resource: &str,
) -> u64 {
    let key = (tensor.to_owned(), resource.to_owned());
    let Some(alloc_id) = copies.remove(&key) else {
        return 0;
    };
    let Some(rec) = allocations.get_mut(&alloc_id) else {
        return 0;
    };
    if rec.2 > 1 {
        rec.2 -= 1;
        return 0;
    }
    let (mem, capacity, _) = allocations.remove(&alloc_id).unwrap();
    if let Some(v) = resident.get_mut(&mem) {
        *v = v.saturating_sub(capacity);
    }
    capacity
}

fn release_state_due(
    leases: &mut Vec<(f64, String, u64)>,
    resident: &mut HashMap<String, u64>,
    at_s: f64,
) {
    let mut kept = Vec::new();
    for (end_s, mem, nbytes) in leases.drain(..) {
        if end_s <= at_s + 1e-15 {
            if let Some(v) = resident.get_mut(&mem) {
                *v = v.saturating_sub(nbytes);
            }
        } else {
            kept.push((end_s, mem, nbytes));
        }
    }
    *leases = kept;
}

fn live_activation_bytes(
    copies: &HashMap<(String, String), String>,
    allocations: &HashMap<String, (String, u64, u32)>,
    activation_ids: &HashSet<String>,
) -> u64 {
    let mut active: HashSet<&str> = HashSet::new();
    for ((tid, rid), alloc_id) in copies {
        if rid == "disk" || !activation_ids.contains(tid) {
            continue;
        }
        active.insert(alloc_id.as_str());
    }
    active
        .into_iter()
        .filter_map(|a| allocations.get(a).map(|(_, cap, _)| *cap))
        .sum()
}

fn attr_f64(v: &streamcompiler_core::AttrValue) -> Option<f64> {
    match v {
        streamcompiler_core::AttrValue::Float(f) => Some(*f),
        streamcompiler_core::AttrValue::Int(i) => Some(*i as f64),
        _ => None,
    }
}

fn attr_u64(v: &streamcompiler_core::AttrValue) -> Option<u64> {
    match v {
        streamcompiler_core::AttrValue::Int(i) if *i >= 0 => Some(*i as u64),
        streamcompiler_core::AttrValue::Float(f) if *f >= 0.0 => Some(*f as u64),
        _ => None,
    }
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
            backend_id: None,
            transfer_backend: None,
            sync_required: false,
            stream_id: None,
            copy_engine_id: None,
            link_id: None,
            io_queue_id: None,
            attributes: Default::default(),
        };
        ExecutableSchedule::new("g", "fp", vec![a], vec![])
    }

    #[test]
    fn deterministic_makespan() {
        let s = simple_schedule();
        let m = MachineModel::cpu_only();
        let r1 = match simulate_schedule(&s, &m).unwrap() {
            SimulationOutcome::Valid(r) => r,
            other => panic!("expected Valid, got {other:?}"),
        };
        let r2 = match simulate_schedule(&s, &m).unwrap() {
            SimulationOutcome::Valid(r) => r,
            other => panic!("expected Valid, got {other:?}"),
        };
        assert!((r1.makespan_s - r2.makespan_s).abs() < 1e-15);
        assert!(r1.simulated);
    }
}
