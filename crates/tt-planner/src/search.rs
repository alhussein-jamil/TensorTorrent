//! Beam search, local search, parallel subset planning, finalist selection.

use crate::problem::{CandidateKernel, PlacementRecord, PlanningProblem};
use crate::score::{analytic_score, comparable_finalist_score};
use crate::{resolve_workers, should_parallelize_subsets};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::time::Instant;

/// Compact partial assignment. Indexed Vecs — no String in hot loops.
#[derive(Clone, Debug)]
pub struct SearchState {
    /// Candidate index into problem.candidates[region] for each placed region in order.
    pub placement_cands: Vec<u16>,
    pub finish: Vec<f64>,
    pub device_free: Vec<f64>,
    pub link_free: HashMap<u32, f64>,
    pub device_busy: Vec<f64>,
    pub link_busy: HashMap<u32, f64>,
    pub live_activation_bytes: Vec<u64>,
    pub peak_bytes: Vec<u64>,
    pub output_device: Vec<u16>,
    pub output_bytes: Vec<u64>,
    pub remaining_consumers: Vec<i32>,
    pub transfer_bytes: u64,
    pub transfer_latency_s: f64,
    pub unmeasured_transfer_count: u32,
    pub host_staged_transfer_count: u32,
}

impl SearchState {
    fn new(n_regions: usize, n_devices: usize, consumers: &[i32]) -> Self {
        Self {
            placement_cands: Vec::with_capacity(n_regions),
            finish: vec![0.0; n_regions],
            device_free: vec![0.0; n_devices],
            link_free: HashMap::new(),
            device_busy: vec![0.0; n_devices],
            link_busy: HashMap::new(),
            live_activation_bytes: vec![0; n_devices],
            peak_bytes: vec![0; n_devices],
            output_device: vec![u16::MAX; n_regions],
            output_bytes: vec![0; n_regions],
            remaining_consumers: consumers.to_vec(),
            transfer_bytes: 0,
            transfer_latency_s: 0.0,
            unmeasured_transfer_count: 0,
            host_staged_transfer_count: 0,
        }
    }

    #[must_use]
    pub fn makespan_s(&self) -> f64 {
        self.finish.iter().copied().fold(0.0f64, f64::max)
    }

    #[must_use]
    pub fn initiation_interval_s(&self) -> f64 {
        let d = self.device_busy.iter().copied().fold(0.0f64, f64::max);
        let l = self.link_busy.values().copied().fold(0.0f64, f64::max);
        d.max(l).max(1e-12)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SearchResult {
    pub placements: Vec<PlacementRecord>,
    pub latency_s: f64,
    pub throughput_per_s: f64,
    pub peak_bytes: HashMap<String, u64>,
    pub transfer_bytes: u64,
    pub transfer_latency_s: f64,
    pub unmeasured_transfer_count: u32,
    pub host_staged_transfer_count: u32,
    pub states_expanded: u64,
    pub states_pruned: u64,
    pub beam_width: usize,
    pub local_improvements: u32,
    pub subset_devices: Vec<String>,
    pub analytic_score: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FinalistPlan {
    pub placements: Vec<PlacementRecord>,
    pub latency_s: f64,
    pub throughput_per_s: f64,
    pub peak_bytes: HashMap<String, u64>,
    pub transfer_bytes: u64,
    pub transfer_latency_s: f64,
    pub unmeasured_transfer_count: u32,
    pub host_staged_transfer_count: u32,
    pub states_expanded: u64,
    pub states_pruned: u64,
    pub subset_devices: Vec<String>,
    pub analytic_score: f64,
    pub search_rank: usize,
    pub placement_signature: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PlanStatistics {
    pub planner_engine: String,
    pub planner_workers_requested: usize,
    pub planner_workers_used: usize,
    pub parallel_search_used: bool,
    pub candidate_subsets: usize,
    pub subsets_searched: usize,
    pub states_expanded: u64,
    pub states_pruned: u64,
    pub beam_width: usize,
    pub local_improvements: u32,
    pub finalists_generated: usize,
    pub finalists_deduplicated: usize,
    pub native_search_s: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PlannerOutput {
    pub finalists: Vec<FinalistPlan>,
    pub statistics: PlanStatistics,
}

struct LinkIntern {
    to_id: HashMap<String, u32>,
    names: Vec<String>,
}

impl LinkIntern {
    fn new() -> Self {
        Self {
            to_id: HashMap::new(),
            names: Vec::new(),
        }
    }

    fn id(&mut self, name: &str) -> u32 {
        if let Some(&id) = self.to_id.get(name) {
            return id;
        }
        let id = self.names.len() as u32;
        self.names.push(name.to_owned());
        self.to_id.insert(name.to_owned(), id);
        id
    }
}

type DomKey = (Vec<u16>, Vec<(u16, u64)>, Vec<(u16, i64)>, Vec<(u32, i64)>);

fn dominance_key(state: &SearchState, device_seq: &[u16]) -> DomKey {
    let live: Vec<(u16, u64)> = state
        .live_activation_bytes
        .iter()
        .enumerate()
        .filter(|(_, &b)| b > 0)
        .map(|(i, &b)| (i as u16, b))
        .collect();
    let device_free: Vec<(u16, i64)> = state
        .device_free
        .iter()
        .enumerate()
        .filter(|(_, &v)| v > 0.0)
        .map(|(i, &v)| (i as u16, (v * 1e12).round() as i64))
        .collect();
    let mut link_free: Vec<(u32, i64)> = state
        .link_free
        .iter()
        .map(|(&k, &v)| (k, (v * 1e12).round() as i64))
        .collect();
    link_free.sort_by_key(|(k, _)| *k);
    (device_seq.to_vec(), live, device_free, link_free)
}

fn extend_state(
    state: &SearchState,
    region_idx: usize,
    cand_idx: u16,
    candidate: &CandidateKernel,
    problem: &PlanningProblem,
    links: &mut LinkIntern,
    allowed_mask: &[bool],
) -> Option<SearchState> {
    if !allowed_mask.get(candidate.device).copied().unwrap_or(false) {
        return None;
    }
    let device = candidate.device;
    let mut next = state.clone();
    let deps = &problem.regions[region_idx].depends_on;
    let mut ready = 0.0f64;
    let mut incoming_remote: u64 = 0;

    for &dep in deps {
        let dep_finish = next.finish[dep];
        let dep_device = next.output_device[dep];
        if dep_device == u16::MAX {
            return None;
        }
        let dep_dev = dep_device as usize;
        let nbytes = problem
            .edge_bytes
            .get(&(dep, region_idx))
            .copied()
            .unwrap_or(next.output_bytes[dep]);
        if dep_dev == device {
            ready = ready.max(dep_finish);
            continue;
        }
        let src_mem = &problem.device_memory[dep_dev];
        let dst_mem = &problem.device_memory[device];
        let estimate = problem
            .machine
            .estimate_transfer(src_mem, dst_mem, nbytes)?;
        let link_id = links.id(&estimate.resource);
        let start = dep_finish.max(next.link_free.get(&link_id).copied().unwrap_or(0.0));
        let end = start + estimate.duration_s;
        next.link_free.insert(link_id, end);
        *next.link_busy.entry(link_id).or_insert(0.0) += estimate.duration_s;
        ready = ready.max(end);
        incoming_remote += nbytes;
        next.transfer_bytes += nbytes;
        next.transfer_latency_s += estimate.duration_s;
        if !estimate.measured {
            next.unmeasured_transfer_count += 1;
        }
        if estimate.host_staged {
            next.host_staged_transfer_count += 1;
        }
    }

    let latency = candidate.estimated_latency_s.max(1e-12);
    let start = ready.max(next.device_free[device]);
    let end = start + latency;

    let workspace = candidate.workspace_bytes;
    let output_bytes = problem.regions[region_idx].output_bytes;
    let state_bytes = problem.regions[region_idx].state_bytes;
    let current_live = next.live_activation_bytes[device];
    let instantaneous = current_live + incoming_remote + state_bytes + output_bytes + workspace;
    let capacity = problem.capacities.get(device).copied().unwrap_or(0);
    if capacity > 0 && instantaneous > capacity {
        return None;
    }

    next.peak_bytes[device] = next.peak_bytes[device].max(instantaneous);
    next.live_activation_bytes[device] = current_live + output_bytes;
    next.output_device[region_idx] = device as u16;
    next.output_bytes[region_idx] = output_bytes;
    if next.remaining_consumers[region_idx] <= 0 {
        next.live_activation_bytes[device] =
            next.live_activation_bytes[device].saturating_sub(output_bytes);
    }

    for &dep in deps {
        if dep >= next.remaining_consumers.len() {
            continue;
        }
        next.remaining_consumers[dep] -= 1;
        if next.remaining_consumers[dep] <= 0 {
            let source_device = next.output_device[dep] as usize;
            if source_device < next.live_activation_bytes.len() {
                next.live_activation_bytes[source_device] = next.live_activation_bytes
                    [source_device]
                    .saturating_sub(next.output_bytes[dep]);
            }
        }
    }

    next.finish[region_idx] = end;
    next.device_free[device] = end;
    next.device_busy[device] += latency;
    next.placement_cands.push(cand_idx);
    Some(next)
}

fn filter_pool(
    pool: &[CandidateKernel],
    allowed_mask: &[bool],
    per_device: usize,
) -> Vec<(u16, CandidateKernel)> {
    let mut by_device: HashMap<usize, Vec<(u16, &CandidateKernel)>> = HashMap::new();
    for (i, c) in pool.iter().enumerate() {
        if !allowed_mask.get(c.device).copied().unwrap_or(false) {
            continue;
        }
        by_device.entry(c.device).or_default().push((i as u16, c));
    }
    let mut selected = Vec::new();
    let mut devices: Vec<usize> = by_device.keys().copied().collect();
    devices.sort_unstable();
    for device in devices {
        let mut ranked = by_device.remove(&device).unwrap_or_default();
        ranked.sort_by(|a, b| {
            a.1.estimated_latency_s
                .partial_cmp(&b.1.estimated_latency_s)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.1.workspace_bytes.cmp(&b.1.workspace_bytes))
                .then_with(|| a.1.dtype.cmp(&b.1.dtype))
                .then_with(|| a.1.kernel_id.cmp(&b.1.kernel_id))
        });
        for (idx, c) in ranked.into_iter().take(per_device.max(1)) {
            selected.push((idx, c.clone()));
        }
    }
    selected
}

fn select_beam(
    mut states: Vec<(f64, SearchState, Vec<u16>)>,
    beam_width: usize,
) -> Vec<SearchState> {
    if states.is_empty() {
        return Vec::new();
    }
    // Deterministic: score, then placement signature.
    states.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.2.cmp(&b.2))
    });
    if states.len() > beam_width {
        // Already fully sorted for determinism; take prefix.
        states.truncate(beam_width);
    }
    states.into_iter().map(|(_, s, _)| s).collect()
}

fn placements_from_state(
    state: &SearchState,
    problem: &PlanningProblem,
    pools: &[Vec<(u16, CandidateKernel)>],
) -> Vec<PlacementRecord> {
    let mut out = Vec::with_capacity(problem.order.len());
    for (step, &region_idx) in problem.order.iter().enumerate() {
        let cand_idx = state.placement_cands[step];
        let candidate = pools[region_idx]
            .iter()
            .find(|(i, _)| *i == cand_idx)
            .map(|(_, c)| c)
            .or_else(|| problem.candidates[region_idx].get(cand_idx as usize))
            .expect("placement candidate");
        let deps: Vec<String> = problem.regions[region_idx]
            .depends_on
            .iter()
            .map(|&d| problem.regions[d].name.clone())
            .collect();
        out.push(PlacementRecord {
            region_id: problem.regions[region_idx].name.clone(),
            device: problem.device_names[candidate.device].clone(),
            backend_id: candidate.backend_id.clone(),
            dtype: candidate.dtype.clone(),
            kernel_id: candidate.kernel_id.clone(),
            estimated_latency_s: candidate.estimated_latency_s.max(1e-12),
            depends_on: deps,
            measured: candidate.measured,
            output_bytes: problem.regions[region_idx].output_bytes,
            state_bytes: problem.regions[region_idx].state_bytes,
            workspace_bytes: candidate.workspace_bytes,
        });
    }
    out
}

fn placement_signature(placements: &[PlacementRecord]) -> String {
    placements
        .iter()
        .map(|p| {
            format!(
                "{}:{}:{}:{}:{}",
                p.region_id, p.device, p.backend_id, p.kernel_id, p.dtype
            )
        })
        .collect::<Vec<_>>()
        .join("|")
}

fn result_from_state(
    state: &SearchState,
    problem: &PlanningProblem,
    pools: &[Vec<(u16, CandidateKernel)>],
    subset_devices: &[String],
    expanded: u64,
    pruned: u64,
    local_improvements: u32,
) -> SearchResult {
    let placements = placements_from_state(state, problem, pools);
    let latency = state.makespan_s();
    let ii = state.initiation_interval_s();
    let saturation = 1.0 / ii;
    let closed = problem.config.target_inflight_requests.max(1) as f64 / latency.max(1e-12);
    let throughput = saturation.min(closed);
    let mut peak = HashMap::new();
    for (i, &bytes) in state.peak_bytes.iter().enumerate() {
        if bytes > 0 {
            peak.insert(problem.device_names[i].clone(), bytes);
        }
    }
    let score = comparable_finalist_score(
        latency,
        throughput,
        &state.peak_bytes,
        &problem.capacities,
        &problem.config,
    );
    SearchResult {
        placements,
        latency_s: latency,
        throughput_per_s: throughput,
        peak_bytes: peak,
        transfer_bytes: state.transfer_bytes,
        transfer_latency_s: state.transfer_latency_s,
        unmeasured_transfer_count: state.unmeasured_transfer_count,
        host_staged_transfer_count: state.host_staged_transfer_count,
        states_expanded: expanded,
        states_pruned: pruned,
        beam_width: problem.config.beam_width,
        local_improvements,
        subset_devices: subset_devices.to_vec(),
        analytic_score: score,
    }
}

/// Search one device subset. Deterministic for identical inputs.
pub fn search_subset(
    problem: &PlanningProblem,
    subset_device_indices: &[usize],
) -> Option<SearchResult> {
    if problem.regions.is_empty() || subset_device_indices.is_empty() {
        return None;
    }
    let n_dev = problem.device_names.len();
    let mut allowed_mask = vec![false; n_dev];
    for &d in subset_device_indices {
        if d < n_dev {
            allowed_mask[d] = true;
        }
    }
    let subset_devices: Vec<String> = subset_device_indices
        .iter()
        .filter(|&&d| d < n_dev)
        .map(|&d| problem.device_names[d].clone())
        .collect();

    let per_device = problem.config.candidates_per_device.max(1);
    let pools: Vec<Vec<(u16, CandidateKernel)>> = problem
        .candidates
        .iter()
        .map(|pool| filter_pool(pool, &allowed_mask, per_device))
        .collect();
    for (region_idx, pool) in pools.iter().enumerate() {
        if problem.order.contains(&region_idx) && pool.is_empty() {
            // Only regions in order matter; check those.
        }
    }
    for &region_idx in &problem.order {
        if pools[region_idx].is_empty() {
            return None;
        }
    }

    let consumers: Vec<i32> = problem.regions.iter().map(|r| r.consumer_count).collect();
    let mut beam = vec![SearchState::new(problem.regions.len(), n_dev, &consumers)];
    let mut states_expanded = 0u64;
    let mut states_pruned = 0u64;
    let beam_width = problem.config.beam_width.max(1);
    let mut links = LinkIntern::new();

    for &region_idx in &problem.order {
        let pool = &pools[region_idx];
        let mut next_states: Vec<(f64, SearchState, Vec<u16>)> = Vec::new();
        for state in &beam {
            for &(cand_idx, ref candidate) in pool {
                states_expanded += 1;
                match extend_state(
                    state,
                    region_idx,
                    cand_idx,
                    candidate,
                    problem,
                    &mut links,
                    &allowed_mask,
                ) {
                    Some(extended) => {
                        let mut device_seq: Vec<u16> =
                            Vec::with_capacity(extended.placement_cands.len());
                        for (step, &ci) in extended.placement_cands.iter().enumerate() {
                            let r = problem.order[step];
                            let c = pools[r]
                                .iter()
                                .find(|(i, _)| *i == ci)
                                .map(|(_, c)| c.device as u16)
                                .unwrap_or(0);
                            device_seq.push(c);
                        }
                        let score = analytic_score(&extended, &problem.config, &problem.capacities);
                        next_states.push((score, extended, device_seq));
                    }
                    None => states_pruned += 1,
                }
            }
        }
        if next_states.is_empty() {
            return None;
        }

        // Dominance pruning.
        let mut dominant: HashMap<DomKey, (f64, SearchState, Vec<u16>)> = HashMap::new();
        for (score, state, device_seq) in next_states {
            let key = dominance_key(&state, &device_seq);
            match dominant.get(&key) {
                Some((prev_score, _, _)) if *prev_score <= score => {}
                _ => {
                    dominant.insert(key, (score, state, device_seq));
                }
            }
        }
        let ranked: Vec<_> = dominant.into_values().collect();
        let before = ranked.len();
        beam = select_beam(ranked, beam_width);
        if before > beam.len() {
            states_pruned += (before - beam.len()) as u64;
        }
    }

    let mut best = beam.into_iter().min_by(|a, b| {
        analytic_score(a, &problem.config, &problem.capacities)
            .partial_cmp(&analytic_score(b, &problem.config, &problem.capacities))
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.placement_cands.cmp(&b.placement_cands))
    })?;

    // Local coordinate descent.
    let mut assignment: Vec<(u16, CandidateKernel)> = Vec::with_capacity(problem.order.len());
    for (step, &region_idx) in problem.order.iter().enumerate() {
        let ci = best.placement_cands[step];
        let cand = pools[region_idx]
            .iter()
            .find(|(i, _)| *i == ci)
            .map(|(_, c)| c.clone())
            .unwrap();
        assignment.push((ci, cand));
    }

    let mut local_improvements = 0u32;
    for _ in 0..problem.config.local_search_iters {
        let mut improved = false;
        for index in 0..problem.order.len() {
            let region_idx = problem.order[index];
            let mut incumbent = assignment[index].clone();
            let mut incumbent_state = best.clone();
            let mut incumbent_score = analytic_score(&best, &problem.config, &problem.capacities);
            for &(alt_idx, ref alternate) in &pools[region_idx] {
                if alt_idx == incumbent.0
                    && alternate.device == incumbent.1.device
                    && alternate.kernel_id == incumbent.1.kernel_id
                    && alternate.dtype == incumbent.1.dtype
                    && alternate.backend_id == incumbent.1.backend_id
                {
                    continue;
                }
                // Replay from start with alternate at index (compact; prefix reuse optional).
                states_expanded += (problem.order.len() - index) as u64;
                let mut trial_assign = assignment.clone();
                trial_assign[index] = (alt_idx, alternate.clone());
                let mut state = SearchState::new(problem.regions.len(), n_dev, &consumers);
                let mut ok = true;
                for (step, &ridx) in problem.order.iter().enumerate() {
                    let (ci, ref c) = trial_assign[step];
                    match extend_state(&state, ridx, ci, c, problem, &mut links, &allowed_mask) {
                        Some(s) => state = s,
                        None => {
                            ok = false;
                            states_pruned += 1;
                            break;
                        }
                    }
                }
                if !ok {
                    continue;
                }
                let score = analytic_score(&state, &problem.config, &problem.capacities);
                if score + 1e-15 < incumbent_score {
                    incumbent = (alt_idx, alternate.clone());
                    incumbent_state = state;
                    incumbent_score = score;
                }
            }
            if incumbent.0 != assignment[index].0
                || incumbent.1.device != assignment[index].1.device
                || incumbent.1.kernel_id != assignment[index].1.kernel_id
                || incumbent.1.dtype != assignment[index].1.dtype
            {
                assignment[index] = incumbent;
                best = incumbent_state;
                local_improvements += 1;
                improved = true;
            }
        }
        if !improved {
            break;
        }
    }

    Some(result_from_state(
        &best,
        problem,
        &pools,
        &subset_devices,
        states_expanded,
        states_pruned,
        local_improvements,
    ))
}

/// Plan across all subsets; return diverse top-K finalists.
pub fn plan_placements(problem: &PlanningProblem) -> PlannerOutput {
    let t0 = Instant::now();
    let workers_req = problem.config.planner_workers;
    let workers = resolve_workers(workers_req);
    let parallel = should_parallelize_subsets(
        problem.subsets.len(),
        problem.region_count(),
        problem.config.beam_width,
        problem.avg_candidates_per_region(),
        problem.config.allow_parallel_subsets,
        workers,
    );
    let workers_used = if parallel {
        workers.min(problem.subsets.len()).max(1)
    } else {
        1
    };

    let results: Vec<Option<SearchResult>> = if parallel && problem.subsets.len() > 1 {
        match rayon::ThreadPoolBuilder::new()
            .num_threads(workers_used)
            .build()
        {
            Ok(pool) => pool.install(|| {
                problem
                    .subsets
                    .par_iter()
                    .map(|subset| search_subset(problem, &subset.device_indices))
                    .collect()
            }),
            Err(_) => problem
                .subsets
                .iter()
                .map(|subset| search_subset(problem, &subset.device_indices))
                .collect(),
        }
    } else {
        problem
            .subsets
            .iter()
            .map(|subset| search_subset(problem, &subset.device_indices))
            .collect()
    };

    let mut expanded = 0u64;
    let mut pruned = 0u64;
    let mut local_improvements = 0u32;
    let mut feasible: Vec<SearchResult> = Vec::new();
    for r in results.into_iter().flatten() {
        expanded += r.states_expanded;
        pruned += r.states_pruned;
        local_improvements += r.local_improvements;
        feasible.push(r);
    }

    // Rank by analytic comparable score, then deterministic placement signature.
    feasible.sort_by(|a, b| {
        a.analytic_score
            .partial_cmp(&b.analytic_score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| {
                placement_signature(&a.placements).cmp(&placement_signature(&b.placements))
            })
            .then_with(|| a.subset_devices.cmp(&b.subset_devices))
    });

    let k = problem.config.finalist_count.max(1);
    let mut finalists = Vec::new();
    let mut seen_sig = std::collections::HashSet::new();
    let mut subset_counts: HashMap<Vec<String>, usize> = HashMap::new();
    let generated = feasible.len();

    // First pass: prefer diversity across subsets when scores are close.
    for (rank, result) in feasible.iter().enumerate() {
        if finalists.len() >= k {
            break;
        }
        let sig = placement_signature(&result.placements);
        if !seen_sig.insert(sig.clone()) {
            continue;
        }
        let subset_key = result.subset_devices.clone();
        let count = subset_counts.entry(subset_key.clone()).or_insert(0);
        // Allow at most ceil(k/2) from same subset when others remain, unless few subsets.
        let subset_cap = (k / 2).max(1);
        if *count >= subset_cap && finalists.len() + 1 < k && feasible.len() > finalists.len() + 1 {
            // Defer — try later if slots remain.
            continue;
        }
        *count += 1;
        finalists.push(FinalistPlan {
            placements: result.placements.clone(),
            latency_s: result.latency_s,
            throughput_per_s: result.throughput_per_s,
            peak_bytes: result.peak_bytes.clone(),
            transfer_bytes: result.transfer_bytes,
            transfer_latency_s: result.transfer_latency_s,
            unmeasured_transfer_count: result.unmeasured_transfer_count,
            host_staged_transfer_count: result.host_staged_transfer_count,
            states_expanded: result.states_expanded,
            states_pruned: result.states_pruned,
            subset_devices: result.subset_devices.clone(),
            analytic_score: result.analytic_score,
            search_rank: rank,
            placement_signature: sig,
        });
    }

    // Fill remaining slots without subset cap.
    if finalists.len() < k {
        seen_sig.clear();
        for f in &finalists {
            seen_sig.insert(f.placement_signature.clone());
        }
        for (rank, result) in feasible.iter().enumerate() {
            if finalists.len() >= k {
                break;
            }
            let sig = placement_signature(&result.placements);
            if !seen_sig.insert(sig.clone()) {
                continue;
            }
            finalists.push(FinalistPlan {
                placements: result.placements.clone(),
                latency_s: result.latency_s,
                throughput_per_s: result.throughput_per_s,
                peak_bytes: result.peak_bytes.clone(),
                transfer_bytes: result.transfer_bytes,
                transfer_latency_s: result.transfer_latency_s,
                unmeasured_transfer_count: result.unmeasured_transfer_count,
                host_staged_transfer_count: result.host_staged_transfer_count,
                states_expanded: result.states_expanded,
                states_pruned: result.states_pruned,
                subset_devices: result.subset_devices.clone(),
                analytic_score: result.analytic_score,
                search_rank: rank,
                placement_signature: sig,
            });
        }
    }

    // Re-number search_rank by finalist order.
    for (i, f) in finalists.iter_mut().enumerate() {
        f.search_rank = i;
    }

    let deduped = generated.saturating_sub(finalists.len());
    PlannerOutput {
        statistics: PlanStatistics {
            planner_engine: "rust".into(),
            planner_workers_requested: workers_req,
            planner_workers_used: workers_used,
            parallel_search_used: parallel,
            candidate_subsets: problem.subsets.len(),
            subsets_searched: problem.subsets.len(),
            states_expanded: expanded,
            states_pruned: pruned,
            beam_width: problem.config.beam_width,
            local_improvements,
            finalists_generated: generated,
            finalists_deduplicated: deduped,
            native_search_s: t0.elapsed().as_secs_f64(),
        },
        finalists,
    }
}
