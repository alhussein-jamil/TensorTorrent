//! PyO3 bindings for the native placement planner.

use crate::machine_py::machine_from_py;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList};
use std::collections::HashMap;
use tt_planner::{
    plan_placements, CandidateKernel, ObjectiveKind, PlanningConfig, PlanningProblem, RegionSpec,
    SubsetSpec,
};
use tt_runtime::MachineModel;

#[pyfunction]
#[pyo3(signature = (problem))]
pub(crate) fn plan_placements_py(
    py: Python<'_>,
    problem: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let planning = planning_problem_from_py(problem)?;
    let output = py.detach(|| plan_placements(&planning));
    planner_output_to_dict(py, &output)
}

fn planning_problem_from_py(obj: &Bound<'_, PyAny>) -> PyResult<PlanningProblem> {
    let config_obj = obj.get_item("config")?;
    let config = planning_config_from_py(&config_obj)?;

    let mut machine = if let Ok(m) = obj.get_item("machine") {
        machine_from_py(Some(&m))?
    } else {
        MachineModel::cpu_only()
    };
    machine.allow_host_staged_transfers = config.allow_host_staged_transfers;

    let device_names: Vec<String> = obj.get_item("device_names")?.extract()?;
    let capacities: Vec<u64> = obj
        .get_item("capacities")?
        .extract::<Vec<i64>>()?
        .into_iter()
        .map(|v| v.max(0) as u64)
        .collect();
    let device_memory: Vec<String> = obj.get_item("device_memory")?.extract()?;

    let regions_list = obj.get_item("regions")?;
    let mut regions = Vec::new();
    for item in regions_list.try_iter()? {
        let r = item?;
        let name: String = r.get_item("name")?.extract()?;
        let depends_on: Vec<usize> = r.get_item("depends_on")?.extract()?;
        let output_bytes: u64 = r.get_item("output_bytes")?.extract::<i64>()?.max(0) as u64;
        let state_bytes: u64 = r.get_item("state_bytes")?.extract::<i64>()?.max(0) as u64;
        let consumer_count: i32 = r.get_item("consumer_count")?.extract()?;
        regions.push(RegionSpec {
            name,
            depends_on,
            output_bytes,
            state_bytes,
            consumer_count,
        });
    }

    let order: Vec<usize> = obj.get_item("order")?.extract()?;

    let candidates_list = obj.get_item("candidates")?;
    let mut candidates: Vec<Vec<CandidateKernel>> = Vec::new();
    for region_cands in candidates_list.try_iter()? {
        let region_cands = region_cands?;
        let mut pool = Vec::new();
        for c in region_cands.try_iter()? {
            let c = c?;
            pool.push(CandidateKernel {
                device: c.get_item("device")?.extract()?,
                backend_id: c.get_item("backend_id")?.extract()?,
                kernel_id: c.get_item("kernel_id")?.extract()?,
                dtype: c.get_item("dtype")?.extract()?,
                estimated_latency_s: c.get_item("estimated_latency_s")?.extract()?,
                workspace_bytes: c.get_item("workspace_bytes")?.extract::<i64>()?.max(0) as u64,
                measured: c
                    .get_item("measured")
                    .ok()
                    .and_then(|v| v.extract().ok())
                    .unwrap_or(false),
            });
        }
        candidates.push(pool);
    }

    let mut edge_bytes: HashMap<(usize, usize), u64> = HashMap::new();
    if let Ok(edges) = obj.get_item("edge_bytes") {
        for item in edges.try_iter()? {
            let item = item?;
            // Each edge: (producer, consumer, nbytes) or dict.
            if let Ok(tup) = item.extract::<(usize, usize, i64)>() {
                edge_bytes.insert((tup.0, tup.1), tup.2.max(0) as u64);
            } else {
                let p: usize = item.get_item("producer")?.extract()?;
                let c: usize = item.get_item("consumer")?.extract()?;
                let n: u64 = item.get_item("nbytes")?.extract::<i64>()?.max(0) as u64;
                edge_bytes.insert((p, c), n);
            }
        }
    }

    let mut subsets = Vec::new();
    let subsets_list = obj.get_item("subsets")?;
    for item in subsets_list.try_iter()? {
        let item = item?;
        let device_indices: Vec<usize> = if let Ok(v) = item.extract::<Vec<usize>>() {
            v
        } else {
            item.get_item("device_indices")?.extract()?
        };
        subsets.push(SubsetSpec { device_indices });
    }

    if regions.is_empty() {
        return Err(PyValueError::new_err("planning problem has no regions"));
    }
    if candidates.len() != regions.len() {
        return Err(PyValueError::new_err(
            "candidates length must match regions length",
        ));
    }

    Ok(PlanningProblem {
        regions,
        order,
        candidates,
        device_names,
        capacities,
        device_memory,
        edge_bytes,
        subsets,
        machine,
        config,
    })
}

fn planning_config_from_py(obj: &Bound<'_, PyAny>) -> PyResult<PlanningConfig> {
    let objective_s: String = obj
        .get_item("objective")
        .ok()
        .and_then(|v| v.extract().ok())
        .unwrap_or_else(|| "latency".into());
    Ok(PlanningConfig {
        objective: ObjectiveKind::parse(&objective_s),
        weight_latency: obj
            .get_item("weight_latency")
            .ok()
            .and_then(|v| v.extract().ok())
            .unwrap_or(1.0),
        weight_throughput: obj
            .get_item("weight_throughput")
            .ok()
            .and_then(|v| v.extract().ok())
            .unwrap_or(0.0),
        weight_memory: obj
            .get_item("weight_memory")
            .ok()
            .and_then(|v| v.extract().ok())
            .unwrap_or(0.0),
        beam_width: obj
            .get_item("beam_width")
            .ok()
            .and_then(|v| v.extract().ok())
            .unwrap_or(64),
        candidates_per_device: obj
            .get_item("candidates_per_device")
            .ok()
            .and_then(|v| v.extract().ok())
            .unwrap_or(2),
        local_search_iters: obj
            .get_item("local_search_iters")
            .ok()
            .and_then(|v| v.extract().ok())
            .unwrap_or(2),
        target_inflight_requests: obj
            .get_item("target_inflight_requests")
            .ok()
            .and_then(|v| v.extract().ok())
            .unwrap_or(1),
        allow_host_staged_transfers: obj
            .get_item("allow_host_staged_transfers")
            .ok()
            .and_then(|v| v.extract().ok())
            .unwrap_or(true),
        vram_budget_bytes: obj.get_item("vram_budget_bytes").ok().and_then(|v| {
            if v.is_none() {
                None
            } else {
                v.extract::<i64>().ok().map(|n| n.max(0) as u64)
            }
        }),
        planner_workers: obj
            .get_item("planner_workers")
            .ok()
            .and_then(|v| v.extract().ok())
            .unwrap_or(0),
        allow_parallel_subsets: obj
            .get_item("allow_parallel_subsets")
            .ok()
            .and_then(|v| v.extract().ok())
            .unwrap_or(true),
        finalist_count: obj
            .get_item("finalist_count")
            .ok()
            .and_then(|v| v.extract().ok())
            .unwrap_or(12),
        per_subset_finalists: obj
            .get_item("per_subset_finalists")
            .ok()
            .and_then(|v| v.extract().ok())
            .unwrap_or(0),
    })
}

fn planner_output_to_dict(
    py: Python<'_>,
    output: &tt_planner::PlannerOutput,
) -> PyResult<Py<PyAny>> {
    let d = PyDict::new(py);
    let stats = PyDict::new(py);
    let s = &output.statistics;
    stats.set_item("planner_engine", &s.planner_engine)?;
    stats.set_item("planner_workers_requested", s.planner_workers_requested)?;
    stats.set_item("planner_workers_used", s.planner_workers_used)?;
    stats.set_item("parallel_search_used", s.parallel_search_used)?;
    stats.set_item("parallel_beam_used", s.parallel_beam_used)?;
    stats.set_item("candidate_subsets", s.candidate_subsets)?;
    stats.set_item("subsets_searched", s.subsets_searched)?;
    stats.set_item("states_expanded", s.states_expanded)?;
    stats.set_item("states_pruned", s.states_pruned)?;
    stats.set_item("beam_width", s.beam_width)?;
    stats.set_item("local_improvements", s.local_improvements)?;
    stats.set_item("finalists_generated", s.finalists_generated)?;
    stats.set_item("finalists_deduplicated", s.finalists_deduplicated)?;
    stats.set_item("native_search_s", s.native_search_s)?;
    d.set_item("statistics", stats)?;

    let finalists = PyList::empty(py);
    for f in &output.finalists {
        let fd = PyDict::new(py);
        let placements = PyList::empty(py);
        for p in &f.placements {
            let pd = PyDict::new(py);
            pd.set_item("region_id", &p.region_id)?;
            pd.set_item("device", &p.device)?;
            pd.set_item("backend_id", &p.backend_id)?;
            pd.set_item("dtype", &p.dtype)?;
            pd.set_item("kernel_id", &p.kernel_id)?;
            pd.set_item("estimated_latency_s", p.estimated_latency_s)?;
            pd.set_item("depends_on", &p.depends_on)?;
            pd.set_item("measured", p.measured)?;
            pd.set_item("output_bytes", p.output_bytes)?;
            pd.set_item("state_bytes", p.state_bytes)?;
            pd.set_item("workspace_bytes", p.workspace_bytes)?;
            placements.append(pd)?;
        }
        fd.set_item("placements", placements)?;
        fd.set_item("latency_s", f.latency_s)?;
        fd.set_item("throughput_per_s", f.throughput_per_s)?;
        fd.set_item("peak_bytes", &f.peak_bytes)?;
        fd.set_item("transfer_bytes", f.transfer_bytes)?;
        fd.set_item("transfer_latency_s", f.transfer_latency_s)?;
        fd.set_item("unmeasured_transfer_count", f.unmeasured_transfer_count)?;
        fd.set_item("host_staged_transfer_count", f.host_staged_transfer_count)?;
        fd.set_item("states_expanded", f.states_expanded)?;
        fd.set_item("states_pruned", f.states_pruned)?;
        fd.set_item("subset_devices", &f.subset_devices)?;
        fd.set_item("analytic_score", f.analytic_score)?;
        fd.set_item("search_rank", f.search_rank)?;
        fd.set_item("placement_signature", &f.placement_signature)?;
        finalists.append(fd)?;
    }
    d.set_item("finalists", finalists)?;
    Ok(d.into())
}
