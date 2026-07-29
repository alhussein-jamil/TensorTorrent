//! Structural validation for executable schedules.

use crate::error::{CoreError, CoreResult};
use crate::ids::InstructionId;
use crate::opcode::Opcode;
use crate::schedule::ExecutableSchedule;
use std::collections::{HashMap, HashSet, VecDeque};

/// Human-readable validation report. Empty `errors` means safe to execute/simulate.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ValidationReport {
    pub errors: Vec<String>,
}

impl ValidationReport {
    #[must_use]
    pub fn ok(&self) -> bool {
        self.errors.is_empty()
    }
}

/// Validate structural invariants. Never reorders or drops instructions.
#[must_use]
pub fn validate_schedule(schedule: &ExecutableSchedule) -> ValidationReport {
    let mut errors = Vec::new();
    let mut by_name: HashMap<&str, &crate::instruction::Instruction> = HashMap::new();

    for inst in &schedule.instructions {
        let name = inst.name.as_str();
        if by_name.contains_key(name) {
            errors.push(format!("duplicate instruction id: {name:?}"));
        } else {
            by_name.insert(name, inst);
        }
    }

    let mut recorded_events: HashSet<&str> = HashSet::new();
    for inst in &schedule.instructions {
        for dep in &inst.depends_on {
            if !by_name.contains_key(dep.as_str()) {
                errors.push(format!(
                    "{:?} depends on unknown instruction {:?}",
                    inst.name.as_str(),
                    dep.as_str()
                ));
            }
        }
        match inst.opcode {
            Opcode::RecordEvent => {
                recorded_events.insert(inst.name.as_str());
            }
            Opcode::Transfer => {
                if inst.source.is_none() || inst.destination.is_none() {
                    errors.push(format!(
                        "transfer {:?} missing source or destination",
                        inst.name.as_str()
                    ));
                }
                if inst.inputs.is_empty() && inst.outputs.is_empty() {
                    errors.push(format!(
                        "transfer {:?} references no tensors",
                        inst.name.as_str()
                    ));
                }
                for tid in inst.inputs.iter().chain(inst.outputs.iter()) {
                    if tid.as_str().is_empty() {
                        errors.push(format!(
                            "transfer {:?} has empty tensor id",
                            inst.name.as_str()
                        ));
                    }
                }
            }
            Opcode::Compute => {
                if inst.resource.as_str().is_empty() {
                    errors.push(format!("compute {:?} missing resource", inst.name.as_str()));
                }
                if inst.executable_ref.is_none() {
                    errors.push(format!(
                        "compute {:?} missing executable_ref",
                        inst.name.as_str()
                    ));
                }
                for tid in inst.inputs.iter().chain(inst.outputs.iter()) {
                    if tid.as_str().is_empty() {
                        errors.push(format!(
                            "compute {:?} has empty tensor id",
                            inst.name.as_str()
                        ));
                    }
                }
            }
            Opcode::Load | Opcode::Prefetch | Opcode::Evict | Opcode::Release => {
                for tid in inst.inputs.iter().chain(inst.outputs.iter()) {
                    if tid.as_str().is_empty() {
                        errors.push(format!(
                            "{} {:?} has empty tensor id",
                            inst.opcode,
                            inst.name.as_str()
                        ));
                    }
                }
                if inst.opcode == Opcode::Load && inst.inputs.is_empty() && inst.outputs.is_empty()
                {
                    errors.push(format!(
                        "load {:?} references no tensors",
                        inst.name.as_str()
                    ));
                }
                if inst.opcode == Opcode::Release && inst.inputs.is_empty() {
                    errors.push(format!(
                        "release {:?} references no tensors",
                        inst.name.as_str()
                    ));
                }
                if inst.opcode == Opcode::Evict && inst.inputs.is_empty() {
                    errors.push(format!(
                        "evict {:?} references no tensors",
                        inst.name.as_str()
                    ));
                }
                if matches!(
                    inst.opcode,
                    Opcode::Load | Opcode::Prefetch | Opcode::Evict | Opcode::Release
                ) {
                    // Structural checks only; exact sizes via validate_tensor_sizes.
                }
            }
            Opcode::WaitEvent => {}
        }
        // Explicit stream / copy-engine / link resources.
        match inst.opcode {
            Opcode::Compute | Opcode::Transfer | Opcode::Load | Opcode::Prefetch
                if inst
                    .stream_id
                    .as_ref()
                    .map(|s| s.as_str().is_empty())
                    .unwrap_or(true) =>
            {
                errors.push(format!(
                    "{:?} {:?} missing stream_id",
                    inst.opcode,
                    inst.name.as_str()
                ));
            }
            _ => {}
        }
        if matches!(inst.opcode, Opcode::Transfer | Opcode::Load | Opcode::Prefetch)
            && inst
                .copy_engine_id
                .as_ref()
                .map(|s| s.is_empty())
                .unwrap_or(true)
        {
            errors.push(format!(
                "{:?} {:?} missing copy_engine_id",
                inst.opcode,
                inst.name.as_str()
            ));
        }
        if inst.opcode == Opcode::Transfer
            && inst.link_id.as_ref().map(|s| s.is_empty()).unwrap_or(true)
        {
            errors.push(format!(
                "transfer {:?} missing link_id",
                inst.name.as_str()
            ));
        }
    }

    // Kahn cycle detection.
    let mut indegree: HashMap<&str, usize> = by_name
        .iter()
        .map(|(n, i)| (*n, i.depends_on.len()))
        .collect();
    let mut dependents: HashMap<&str, Vec<&str>> =
        by_name.keys().map(|n| (*n, Vec::new())).collect();
    for (name, inst) in &by_name {
        for dep in &inst.depends_on {
            if let Some(list) = dependents.get_mut(dep.as_str()) {
                list.push(*name);
            }
        }
    }
    let mut ready: VecDeque<&str> = indegree
        .iter()
        .filter_map(|(n, d)| if *d == 0 { Some(*n) } else { None })
        .collect();
    let mut order = Vec::new();
    while let Some(name) = ready.pop_front() {
        order.push(name);
        if let Some(nexts) = dependents.get(name) {
            for nxt in nexts {
                if let Some(deg) = indegree.get_mut(nxt) {
                    *deg -= 1;
                    if *deg == 0 {
                        ready.push_back(nxt);
                    }
                }
            }
        }
    }
    if order.len() != by_name.len() {
        let cyclic: Vec<_> = by_name
            .keys()
            .filter(|n| !order.contains(n))
            .map(|s| (*s).to_owned())
            .collect();
        errors.push(format!("dependency cycle involves: {cyclic:?}"));
        return ValidationReport { errors };
    }

    // WaitEvent must reference a RecordEvent ancestor when waits_for is set.
    for inst in &schedule.instructions {
        if inst.opcode != Opcode::WaitEvent {
            continue;
        }
        let waits_for = inst.attr_str("waits_for").map(str::to_owned).or_else(|| {
            inst.depends_on.first().and_then(|d| {
                by_name.get(d.as_str()).and_then(|dep| {
                    if dep.opcode == Opcode::RecordEvent {
                        Some(dep.name.as_str().to_owned())
                    } else {
                        None
                    }
                })
            })
        });
        if let Some(wf) = waits_for {
            if !recorded_events.contains(wf.as_str()) && !by_name.contains_key(wf.as_str()) {
                errors.push(format!(
                    "wait {:?} references unknown record event {wf:?}",
                    inst.name.as_str()
                ));
            }
        }
    }

    ValidationReport { errors }
}

fn reject_incomplete_sizes(inst: &crate::instruction::Instruction, errors: &mut Vec<String>) {
    if !matches!(
        inst.opcode,
        Opcode::Prefetch
            | Opcode::Load
            | Opcode::Transfer
            | Opcode::Compute
            | Opcode::Evict
            | Opcode::Release
    ) {
        return;
    }
    let mut seen = std::collections::BTreeSet::new();
    let mut tensors = Vec::new();
    for t in inst.inputs.iter().chain(inst.outputs.iter()) {
        if seen.insert(t.as_str()) {
            tensors.push(t.as_str());
        }
    }
    if tensors.is_empty() {
        return;
    }
    let sizes = inst.tensor_nbytes();
    for tensor in &tensors {
        if sizes.contains_key(*tensor) {
            continue;
        }
        if tensors.len() == 1 {
            continue;
        }
        errors.push(format!(
            "instruction {:?} lacks exact tensor_nbytes for {tensor:?}",
            inst.name.as_str()
        ));
    }
}

/// Require exact per-tensor sizes when an instruction names multiple tensors.
#[must_use]
pub fn validate_tensor_sizes(schedule: &ExecutableSchedule) -> ValidationReport {
    let mut errors = Vec::new();
    for inst in &schedule.instructions {
        reject_incomplete_sizes(inst, &mut errors);
    }
    ValidationReport { errors }
}

/// Fail with [`CoreError::Validation`] when the schedule is invalid.
pub fn assert_schedule_valid(schedule: &ExecutableSchedule) -> CoreResult<()> {
    let report = validate_schedule(schedule);
    if report.ok() {
        Ok(())
    } else {
        Err(CoreError::Validation(report.errors.join("; ")))
    }
}

/// Topological order of instruction names, or error on cycle.
pub fn topological_order(schedule: &ExecutableSchedule) -> CoreResult<Vec<InstructionId>> {
    let report = validate_schedule(schedule);
    if !report.ok() {
        return Err(CoreError::Validation(report.errors.join("; ")));
    }
    let mut by_name: HashMap<&str, &crate::instruction::Instruction> = HashMap::new();
    for inst in &schedule.instructions {
        by_name.insert(inst.name.as_str(), inst);
    }
    let mut indegree: HashMap<&str, usize> = by_name
        .iter()
        .map(|(n, i)| (*n, i.depends_on.len()))
        .collect();
    let mut dependents: HashMap<&str, Vec<&str>> =
        by_name.keys().map(|n| (*n, Vec::new())).collect();
    for (name, inst) in &by_name {
        for dep in &inst.depends_on {
            if let Some(list) = dependents.get_mut(dep.as_str()) {
                list.push(*name);
            }
        }
    }
    let mut ready: VecDeque<&str> = indegree
        .iter()
        .filter_map(|(n, d)| if *d == 0 { Some(*n) } else { None })
        .collect();
    // Stable: sort ready set by name when multiple roots.
    let mut order = Vec::new();
    while !ready.is_empty() {
        let mut batch: Vec<&str> = ready.drain(..).collect();
        batch.sort_unstable();
        for name in batch {
            order.push(InstructionId::new(name));
            if let Some(nexts) = dependents.get(name) {
                for nxt in nexts {
                    if let Some(deg) = indegree.get_mut(nxt) {
                        *deg -= 1;
                        if *deg == 0 {
                            ready.push_back(nxt);
                        }
                    }
                }
            }
        }
    }
    Ok(order)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::{InstructionId, RegionId, ResourceId, TensorId};
    use crate::instruction::{AttrValue, Instruction, MemoryTier};
    use indexmap::IndexMap;

    fn compute(name: &str, deps: &[&str]) -> Instruction {
        Instruction {
            opcode: Opcode::Compute,
            name: InstructionId::new(name),
            resource: ResourceId::new("cpu"),
            depends_on: deps.iter().map(|d| InstructionId::new(*d)).collect(),
            inputs: vec![TensorId::new("x")],
            outputs: vec![TensorId::new("y")],
            nbytes: 64,
            memory_tier: MemoryTier::SystemRam,
            predicted_duration_s: 0.01,
            executable_ref: Some(RegionId::new("r0")),
            source: None,
            destination: None,
            backend_id: Some("cpu".into()),
            transfer_backend: None,
            sync_required: false,
            stream_id: Some(crate::ids::StreamId::new("cpu::compute")),
            copy_engine_id: None,
            link_id: None,
            attributes: IndexMap::new(),
        }
    }

    #[test]
    fn accepts_linear_dag() {
        let schedule = ExecutableSchedule::new(
            "g",
            "fp",
            vec![compute("a", &[]), compute("b", &["a"])],
            vec![],
        );
        assert!(validate_schedule(&schedule).ok());
    }

    #[test]
    fn rejects_cycle() {
        let schedule = ExecutableSchedule::new(
            "g",
            "fp",
            vec![compute("a", &["b"]), compute("b", &["a"])],
            vec![],
        );
        let report = validate_schedule(&schedule);
        assert!(!report.ok());
        assert!(report.errors.iter().any(|e| e.contains("cycle")));
    }

    #[test]
    fn rejects_duplicate_ids() {
        let schedule = ExecutableSchedule::new(
            "g",
            "fp",
            vec![compute("a", &[]), compute("a", &[])],
            vec![],
        );
        assert!(!validate_schedule(&schedule).ok());
    }

    #[test]
    fn rejects_multi_tensor_without_sizes() {
        let inst = Instruction {
            opcode: Opcode::Compute,
            name: InstructionId::new("c"),
            resource: ResourceId::new("cpu"),
            depends_on: vec![],
            inputs: vec![TensorId::new("a"), TensorId::new("b")],
            outputs: vec![TensorId::new("c")],
            nbytes: 100,
            memory_tier: MemoryTier::SystemRam,
            predicted_duration_s: 0.0,
            executable_ref: Some(RegionId::new("r")),
            source: None,
            destination: None,
            backend_id: None,
            transfer_backend: None,
            sync_required: false,
            stream_id: Some(crate::ids::StreamId::new("cpu::compute")),
            copy_engine_id: None,
            link_id: None,
            attributes: IndexMap::new(),
        };
        let schedule = ExecutableSchedule::new("g", "fp", vec![inst], vec![]);
        assert!(validate_schedule(&schedule).ok());
        let report = validate_tensor_sizes(&schedule);
        assert!(!report.ok());
        assert!(report.errors.iter().any(|e| e.contains("tensor_nbytes")));
    }

    #[test]
    fn json_round_trip() {
        let mut attrs = IndexMap::new();
        attrs.insert(
            "tensor_nbytes".into(),
            AttrValue::IntMap([("x".into(), 8i64), ("y".into(), 16)].into_iter().collect()),
        );
        let mut inst = compute("a", &[]);
        inst.attributes = attrs;
        let schedule = ExecutableSchedule::new("g", "fp", vec![inst], vec!["n".into()]);
        let bytes = schedule.to_json_bytes().unwrap();
        let back = ExecutableSchedule::from_json_bytes(&bytes).unwrap();
        assert_eq!(schedule, back);
    }
}
