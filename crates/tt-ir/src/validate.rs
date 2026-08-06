//! Structural validation for executable schedules.

use crate::error::{CoreError, CoreResult};
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
        if matches!(
            inst.opcode,
            Opcode::Transfer | Opcode::Load | Opcode::Prefetch
        ) && inst
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
            errors.push(format!("transfer {:?} missing link_id", inst.name.as_str()));
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

    // Memoized ancestor closure for release / transfer-ordering checks.
    let mut ancestor_cache: HashMap<&str, HashSet<&str>> = HashMap::new();
    fn ancestors<'a>(
        name: &'a str,
        by_name: &HashMap<&'a str, &'a crate::instruction::Instruction>,
        cache: &mut HashMap<&'a str, HashSet<&'a str>>,
    ) -> HashSet<&'a str> {
        if let Some(hit) = cache.get(name) {
            return hit.clone();
        }
        let mut seen: HashSet<&str> = HashSet::new();
        let mut stack: Vec<&str> = by_name
            .get(name)
            .map(|i| i.depends_on.iter().map(|d| d.as_str()).collect())
            .unwrap_or_default();
        while let Some(cur) = stack.pop() {
            if !seen.insert(cur) || !by_name.contains_key(cur) {
                continue;
            }
            if let Some(hit) = cache.get(cur) {
                seen.extend(hit.iter().copied());
            } else if let Some(inst) = by_name.get(cur) {
                stack.extend(inst.depends_on.iter().map(|d| d.as_str()));
            }
        }
        cache.insert(name, seen.clone());
        seen
    }

    // (tensor, destination resource) → transfer/wait completion instruction.
    let mut transfer_completion_for: HashMap<(String, String), &str> = HashMap::new();
    for (name, inst) in &by_name {
        if inst.opcode == Opcode::WaitEvent {
            let dest = inst.resource.as_str();
            for out in &inst.inputs {
                transfer_completion_for
                    .entry((out.as_str().to_owned(), dest.to_owned()))
                    .or_insert(*name);
            }
        } else if inst.opcode == Opcode::Transfer {
            let dest = inst
                .destination
                .as_ref()
                .map(|d| d.as_str())
                .unwrap_or("")
                .to_owned();
            let tensors: Vec<&str> = if !inst.outputs.is_empty() {
                inst.outputs.iter().map(|t| t.as_str()).collect()
            } else {
                inst.inputs.iter().map(|t| t.as_str()).collect()
            };
            for out in tensors {
                transfer_completion_for
                    .entry((out.to_owned(), dest.clone()))
                    .or_insert(*name);
            }
        }
    }

    let mut consumers_by_tensor: HashMap<&str, Vec<&str>> = HashMap::new();
    let mut producers_by_tensor: HashMap<&str, Vec<&str>> = HashMap::new();
    for (name, inst) in &by_name {
        for value in &inst.inputs {
            consumers_by_tensor
                .entry(value.as_str())
                .or_default()
                .push(*name);
        }
        match inst.opcode {
            Opcode::Compute => {
                for value in &inst.outputs {
                    producers_by_tensor
                        .entry(value.as_str())
                        .or_default()
                        .push(*name);
                }
            }
            Opcode::Transfer | Opcode::Load => {
                for value in inst.outputs.iter().chain(inst.inputs.iter()) {
                    producers_by_tensor
                        .entry(value.as_str())
                        .or_default()
                        .push(*name);
                }
            }
            _ => {}
        }
    }

    for (name, inst) in &by_name {
        if inst.opcode == Opcode::WaitEvent {
            let mut waits_for = inst.attr_str("waits_for").map(str::to_owned);
            if waits_for.as_ref().map(|s| s.is_empty()).unwrap_or(true) {
                waits_for = inst.depends_on.first().and_then(|d| {
                    by_name.get(d.as_str()).and_then(|dep| {
                        (dep.opcode == Opcode::RecordEvent).then(|| dep.name.as_str().to_owned())
                    })
                });
            }
            match waits_for.as_deref() {
                Some(wf) if !wf.is_empty() => {
                    if !recorded_events.contains(wf) && !by_name.contains_key(wf) {
                        errors.push(format!(
                            "wait {:?} references unknown event {wf:?}",
                            inst.name.as_str()
                        ));
                    } else if let Some(target) = by_name.get(wf) {
                        if target.opcode != Opcode::RecordEvent {
                            errors.push(format!(
                                "wait {:?} waits for non-RecordEvent {wf:?}",
                                inst.name.as_str()
                            ));
                        } else if !recorded_events.contains(wf) {
                            errors.push(format!(
                                "wait {:?} for event that is never recorded: {wf:?}",
                                inst.name.as_str()
                            ));
                        }
                    } else if !recorded_events.contains(wf) {
                        errors.push(format!(
                            "wait {:?} for event that is never recorded: {wf:?}",
                            inst.name.as_str()
                        ));
                    }
                }
                _ => {
                    errors.push(format!(
                        "wait {:?} has no RecordEvent target",
                        inst.name.as_str()
                    ));
                }
            }
        }
        if inst.opcode == Opcode::Release {
            let release_anc = ancestors(name, &by_name, &mut ancestor_cache);
            for value in &inst.inputs {
                for consumer in consumers_by_tensor
                    .get(value.as_str())
                    .into_iter()
                    .flatten()
                {
                    if *consumer == *name {
                        continue;
                    }
                    // Every reader of the tensor must precede this Release in the
                    // dependency DAG (not merely in one Kahn tie-break order).
                    if !release_anc.contains(consumer) {
                        errors.push(format!(
                            "release {:?} of {:?} happens before consumer {consumer:?}",
                            inst.name.as_str(),
                            value.as_str()
                        ));
                    }
                }
            }
        }
        if inst.opcode == Opcode::Compute {
            let compute_anc = ancestors(name, &by_name, &mut ancestor_cache);
            let resource = inst.resource.as_str();
            for value in &inst.inputs {
                let tid = value.as_str();
                let completion = transfer_completion_for
                    .get(&(tid.to_owned(), resource.to_owned()))
                    .or_else(|| transfer_completion_for.get(&(tid.to_owned(), String::new())))
                    .copied();
                if let Some(completion) = completion {
                    if !compute_anc.contains(completion) {
                        errors.push(format!(
                            "compute {:?} reads {tid:?} without depending on transfer completion {completion:?}",
                            inst.name.as_str()
                        ));
                    }
                }
                let producers = producers_by_tensor
                    .get(tid)
                    .map(|v| v.as_slice())
                    .unwrap_or(&[]);
                let has_activation_producer = producers.iter().any(|pname| {
                    by_name
                        .get(pname)
                        .is_some_and(|p| p.opcode == Opcode::Compute)
                });
                if !has_activation_producer {
                    continue;
                }
                let local_ok = producers.iter().any(|pname| {
                    let Some(pinst) = by_name.get(pname) else {
                        return false;
                    };
                    match pinst.opcode {
                        Opcode::Compute => pinst.resource.as_str() == resource,
                        Opcode::Transfer => pinst
                            .destination
                            .as_ref()
                            .is_some_and(|d| d.as_str() == resource),
                        Opcode::Load => {
                            let dest = pinst
                                .destination
                                .as_ref()
                                .map(|d| d.as_str())
                                .unwrap_or_else(|| pinst.resource.as_str());
                            dest == resource
                        }
                        _ => false,
                    }
                });
                if !local_ok {
                    errors.push(format!(
                        "compute {:?} requires copy of {tid:?} on {resource:?} \
                         but schedule only produces it elsewhere (no silent reuse)",
                        inst.name.as_str()
                    ));
                }
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
            io_queue_id: None,
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

    fn transfer(name: &str, src: &str, dst: &str, tensor: &str) -> Instruction {
        Instruction {
            opcode: Opcode::Transfer,
            name: InstructionId::new(name),
            resource: ResourceId::new("copy_engine"),
            depends_on: vec![],
            inputs: vec![TensorId::new(tensor)],
            outputs: vec![TensorId::new(tensor)],
            nbytes: 8,
            memory_tier: MemoryTier::SystemRam,
            predicted_duration_s: 0.0,
            executable_ref: None,
            source: Some(ResourceId::new(src)),
            destination: Some(ResourceId::new(dst)),
            backend_id: None,
            transfer_backend: None,
            sync_required: false,
            stream_id: Some(crate::ids::StreamId::new("copy_engine::copy0")),
            copy_engine_id: Some("copy_engine::copy0".into()),
            link_id: Some(format!("{src}->{dst}")),
            io_queue_id: None,
            attributes: IndexMap::new(),
        }
    }

    #[test]
    fn rejects_release_before_consumer() {
        let producer = Instruction {
            outputs: vec![TensorId::new("t0")],
            inputs: vec![],
            ..compute("producer", &[])
        };
        let release = Instruction {
            opcode: Opcode::Release,
            name: InstructionId::new("release::t0"),
            resource: ResourceId::new("cpu"),
            depends_on: vec![InstructionId::new("producer")],
            inputs: vec![TensorId::new("t0")],
            outputs: vec![],
            nbytes: 0,
            memory_tier: MemoryTier::SystemRam,
            predicted_duration_s: 0.0,
            executable_ref: None,
            source: None,
            destination: None,
            backend_id: None,
            transfer_backend: None,
            sync_required: false,
            stream_id: Some(crate::ids::StreamId::new("cpu::lifetime")),
            copy_engine_id: None,
            link_id: None,
            io_queue_id: None,
            attributes: IndexMap::new(),
        };
        let consumer = Instruction {
            inputs: vec![TensorId::new("t0")],
            outputs: vec![TensorId::new("y")],
            ..compute("consumer", &["producer"])
        };
        let schedule =
            ExecutableSchedule::new("g", "fp", vec![producer, release, consumer], vec![]);
        let report = validate_schedule(&schedule);
        assert!(report
            .errors
            .iter()
            .any(|e| e.contains("happens before consumer")));
    }

    #[test]
    fn rejects_compute_without_transfer_dep() {
        let xfer = transfer("transfer::t0", "cpu_a", "cpu_b", "t0");
        let consumer = Instruction {
            resource: ResourceId::new("cpu_b"),
            inputs: vec![TensorId::new("t0")],
            outputs: vec![TensorId::new("y")],
            stream_id: Some(crate::ids::StreamId::new("cpu_b::compute")),
            ..compute("consumer", &[])
        };
        let schedule = ExecutableSchedule::new("g", "fp", vec![xfer, consumer], vec![]);
        let report = validate_schedule(&schedule);
        assert!(report
            .errors
            .iter()
            .any(|e| e.contains("without depending on transfer completion")));
    }

    #[test]
    fn rejects_remote_activation_without_copy() {
        let producer = Instruction {
            resource: ResourceId::new("cpu_a"),
            outputs: vec![TensorId::new("t0")],
            inputs: vec![],
            stream_id: Some(crate::ids::StreamId::new("cpu_a::compute")),
            ..compute("compute::a", &[])
        };
        let consumer = Instruction {
            resource: ResourceId::new("cpu_b"),
            inputs: vec![TensorId::new("t0")],
            outputs: vec![TensorId::new("y")],
            stream_id: Some(crate::ids::StreamId::new("cpu_b::compute")),
            ..compute("compute::b", &[])
        };
        let schedule = ExecutableSchedule::new("g", "fp", vec![producer, consumer], vec![]);
        let report = validate_schedule(&schedule);
        assert!(report
            .errors
            .iter()
            .any(|e| e.contains("only produces it elsewhere")));
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
            io_queue_id: None,
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
