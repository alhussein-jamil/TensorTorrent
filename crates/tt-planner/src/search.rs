//! Beam search, local search, parallel subset planning, finalist selection.

use crate::problem::{CandidateKernel, PlacementRecord, PlanningProblem};
use crate::score::{analytic_score, comparable_finalist_score};
use crate::{
    beam_parallelism_possible, resolve_workers, should_parallelize_beam, should_parallelize_subsets,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::time::Instant;

/// Compact partial assignment. Indexed Vecs — no String / HashMap in hot loops.
#[derive(Clone, Debug)]
pub struct SearchState {
    /// Candidate index into problem.candidates[region] for each placed region in order.
    pub placement_cands: Vec<u16>,
    pub finish: Vec<f64>,
    pub device_free: Vec<f64>,
    /// Link free-time indexed by interned link id (dense; grows with LinkIntern).
    pub link_free: Vec<f64>,
    pub device_busy: Vec<f64>,
    pub link_busy: Vec<f64>,
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
    pub fn new(n_regions: usize, n_devices: usize, consumers: &[i32]) -> Self {
        Self {
            placement_cands: Vec::with_capacity(n_regions),
            finish: vec![0.0; n_regions],
            device_free: vec![0.0; n_devices],
            link_free: Vec::new(),
            device_busy: vec![0.0; n_devices],
            link_busy: Vec::new(),
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

    fn ensure_link_slots(&mut self, link_id: u32) {
        let need = link_id as usize + 1;
        if self.link_free.len() < need {
            self.link_free.resize(need, 0.0);
            self.link_busy.resize(need, 0.0);
        }
    }

    #[must_use]
    pub fn makespan_s(&self) -> f64 {
        self.finish.iter().copied().fold(0.0f64, f64::max)
    }

    #[must_use]
    pub fn initiation_interval_s(&self) -> f64 {
        let d = self.device_busy.iter().copied().fold(0.0f64, f64::max);
        let l = self.link_busy.iter().copied().fold(0.0f64, f64::max);
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
    /// Rank among analytically scored candidates before diversity/finalist selection.
    pub analytic_rank: usize,
    /// Position in the shortlist passed to DES (0 = first finalist).
    pub finalist_rank: usize,
    /// Alias of `analytic_rank` for older consumers.
    pub search_rank: usize,
    pub placement_signature: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PlanStatistics {
    pub planner_engine: String,
    pub planner_workers_requested: usize,
    /// Resolved worker budget (`0` → available parallelism).
    pub planner_workers_available: usize,
    /// Effective workers that participated (1 when search stayed serial).
    pub planner_workers_used: usize,
    /// Local Rayon pool size actually built (1 = no multi-thread pool).
    pub planner_pool_threads: usize,
    pub parallel_search_used: bool,
    pub parallel_beam_used: bool,
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

/// Interned link resource names → dense ids (shared across a subset search).
///
/// After [`seed_link_intern`], lookups are immutable so beam expand can share
/// one map across Rayon chunks with no clone.
#[derive(Clone, Default)]
pub(crate) struct LinkIntern {
    to_id: HashMap<String, u32>,
    names: Vec<String>,
}

impl LinkIntern {
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert-or-lookup (seeding / serial mutate path).
    pub fn intern(&mut self, name: &str) -> u32 {
        if let Some(&id) = self.to_id.get(name) {
            return id;
        }
        let id = self.names.len() as u32;
        self.names.push(name.to_owned());
        self.to_id.insert(name.to_owned(), id);
        id
    }

    /// Immutable lookup after seeding. Returns `None` if seed missed a name.
    #[must_use]
    pub fn id(&self, name: &str) -> Option<u32> {
        self.to_id.get(name).copied()
    }
}

/// Pre-register every link resource the machine can emit so parallel expand
/// never needs to invent new ids (keeps dense Vec slots consistent).
pub(crate) fn seed_link_intern(problem: &PlanningProblem, links: &mut LinkIntern) {
    for link in &problem.machine.links {
        if !link.id.is_empty() {
            links.intern(&link.id);
        }
        links.intern(&format!("{}->{}", link.source, link.destination));
        links.intern(&format!("{}->{}", link.destination, link.source));
    }
    let mems = &problem.device_memory;
    for a in mems {
        for b in mems {
            if let Some(est) = problem.machine.estimate_transfer(a, b, 1024) {
                links.intern(&est.resource);
            }
        }
    }
}

#[cfg(test)]
pub(crate) fn filter_pool_public(
    pool: &[CandidateKernel],
    allowed_mask: &[bool],
    per_device: usize,
) -> Vec<(u16, CandidateKernel)> {
    filter_pool(pool, allowed_mask, per_device)
}

#[cfg(test)]
pub(crate) fn extend_state_public(
    state: &SearchState,
    region_idx: usize,
    cand_idx: u16,
    candidate: &CandidateKernel,
    problem: &PlanningProblem,
    links: &LinkIntern,
    allowed_mask: &[bool],
) -> Option<SearchState> {
    extend_state(
        state,
        region_idx,
        cand_idx,
        candidate,
        problem,
        links,
        allowed_mask,
    )
}

#[cfg(test)]
pub(crate) fn replay_assignment_public(
    assignment: &[(u16, CandidateKernel)],
    problem: &PlanningProblem,
    links: &LinkIntern,
    allowed_mask: &[bool],
    consumers: &[i32],
    from_step: usize,
    prefix: Option<&SearchState>,
) -> Option<SearchState> {
    replay_assignment(
        assignment,
        problem,
        links,
        allowed_mask,
        consumers,
        from_step,
        prefix,
    )
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
    let link_free: Vec<(u32, i64)> = state
        .link_free
        .iter()
        .enumerate()
        .filter(|(_, &v)| v > 0.0)
        .map(|(i, &v)| (i as u32, (v * 1e12).round() as i64))
        .collect();
    (device_seq.to_vec(), live, device_free, link_free)
}

fn extend_state(
    state: &SearchState,
    region_idx: usize,
    cand_idx: u16,
    candidate: &CandidateKernel,
    problem: &PlanningProblem,
    links: &LinkIntern,
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
        let link_id = links.id(&estimate.resource)?;
        next.ensure_link_slots(link_id);
        let lid = link_id as usize;
        let start = dep_finish.max(next.link_free[lid]);
        let end = start + estimate.duration_s;
        next.link_free[lid] = end;
        next.link_busy[lid] += estimate.duration_s;
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

fn beam_cmp(a: &(f64, SearchState, Vec<u16>), b: &(f64, SearchState, Vec<u16>)) -> Ordering {
    a.0.partial_cmp(&b.0)
        .unwrap_or(Ordering::Equal)
        .then_with(|| a.2.cmp(&b.2))
        .then_with(|| a.1.placement_cands.cmp(&b.1.placement_cands))
}

fn select_beam(
    mut states: Vec<(f64, SearchState, Vec<u16>)>,
    beam_width: usize,
) -> Vec<SearchState> {
    if states.is_empty() {
        return Vec::new();
    }
    let width = beam_width.max(1);
    if states.len() > width * 4 {
        // Partial select then sort survivors — same deterministic order as full sort.
        states.select_nth_unstable_by(width - 1, beam_cmp);
        states.truncate(width);
    }
    states.sort_by(beam_cmp);
    if states.len() > width {
        states.truncate(width);
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

fn finalist_from_result(
    result: &SearchResult,
    analytic_rank: usize,
    finalist_rank: usize,
    signature: String,
) -> FinalistPlan {
    FinalistPlan {
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
        analytic_rank,
        finalist_rank,
        search_rank: analytic_rank,
        placement_signature: signature,
    }
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

fn device_seq_of(
    state: &SearchState,
    problem: &PlanningProblem,
    pools: &[Vec<(u16, CandidateKernel)>],
) -> Vec<u16> {
    let mut device_seq = Vec::with_capacity(state.placement_cands.len());
    for (step, &ci) in state.placement_cands.iter().enumerate() {
        let r = problem.order[step];
        let d = pools[r]
            .iter()
            .find(|(i, _)| *i == ci)
            .map(|(_, c)| c.device as u16)
            .unwrap_or(0);
        device_seq.push(d);
    }
    device_seq
}

fn replay_assignment(
    assignment: &[(u16, CandidateKernel)],
    problem: &PlanningProblem,
    links: &LinkIntern,
    allowed_mask: &[bool],
    consumers: &[i32],
    from_step: usize,
    prefix: Option<&SearchState>,
) -> Option<SearchState> {
    let n_dev = problem.device_names.len();
    let mut state = if from_step == 0 || prefix.is_none() {
        SearchState::new(problem.regions.len(), n_dev, consumers)
    } else {
        prefix.cloned().unwrap()
    };
    let start = if from_step == 0 || prefix.is_none() {
        0
    } else {
        from_step
    };
    // Prefix stores state *before* placing order[from_step].
    for (step, &ridx) in problem.order.iter().enumerate().skip(start) {
        let (ci, ref c) = assignment[step];
        state = extend_state(&state, ridx, ci, c, problem, links, allowed_mask)?;
    }
    Some(state)
}

#[allow(clippy::too_many_arguments)]
fn local_search_improve(
    mut best: SearchState,
    pools: &[Vec<(u16, CandidateKernel)>],
    problem: &PlanningProblem,
    links: &LinkIntern,
    allowed_mask: &[bool],
    consumers: &[i32],
    states_expanded: &mut u64,
    states_pruned: &mut u64,
) -> (SearchState, u32) {
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
        // Prefix cache: state before each step for incremental replay.
        let mut prefixes: Vec<SearchState> = Vec::with_capacity(problem.order.len() + 1);
        prefixes.push(SearchState::new(
            problem.regions.len(),
            problem.device_names.len(),
            consumers,
        ));
        {
            let mut cursor = prefixes[0].clone();
            for (step, &ridx) in problem.order.iter().enumerate() {
                let (ci, ref c) = assignment[step];
                match extend_state(&cursor, ridx, ci, c, problem, links, allowed_mask) {
                    Some(next) => {
                        cursor = next;
                        prefixes.push(cursor.clone());
                    }
                    None => {
                        prefixes.clear();
                        break;
                    }
                }
            }
        }

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
                *states_expanded += (problem.order.len() - index) as u64;
                let mut trial_assign = assignment.clone();
                trial_assign[index] = (alt_idx, alternate.clone());
                let prefix = if prefixes.len() > index {
                    Some(&prefixes[index])
                } else {
                    None
                };
                match replay_assignment(
                    &trial_assign,
                    problem,
                    links,
                    allowed_mask,
                    consumers,
                    index,
                    prefix,
                ) {
                    Some(state) => {
                        let score = analytic_score(&state, &problem.config, &problem.capacities);
                        if score + 1e-15 < incumbent_score {
                            incumbent = (alt_idx, alternate.clone());
                            incumbent_state = state;
                            incumbent_score = score;
                        }
                    }
                    None => *states_pruned += 1,
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
                // Invalidate prefixes; rebuild next outer iteration.
                prefixes.clear();
            }
        }
        if !improved {
            break;
        }
    }
    (best, local_improvements)
}

/// Search one device subset; return multiple distinct high-quality terminals.
///
/// Deterministic for identical inputs (worker count must not change ranking).
pub fn search_subset(
    problem: &PlanningProblem,
    subset_device_indices: &[usize],
) -> Vec<SearchResult> {
    search_subset_ex(problem, subset_device_indices, 1).0
}

/// Like [`search_subset`], also reports whether intra-subset beam expand used Rayon.
pub fn search_subset_ex(
    problem: &PlanningProblem,
    subset_device_indices: &[usize],
    workers: usize,
) -> (Vec<SearchResult>, bool) {
    if problem.regions.is_empty() || subset_device_indices.is_empty() {
        return (Vec::new(), false);
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
    for &region_idx in &problem.order {
        if pools[region_idx].is_empty() {
            return (Vec::new(), false);
        }
    }

    let consumers: Vec<i32> = problem.regions.iter().map(|r| r.consumer_count).collect();
    let mut beam = vec![SearchState::new(problem.regions.len(), n_dev, &consumers)];
    let mut states_expanded = 0u64;
    let mut states_pruned = 0u64;
    let beam_width = problem.config.beam_width.max(1);
    let mut links = LinkIntern::new();
    seed_link_intern(problem, &mut links);
    let mut parallel_beam_used = false;

    for &region_idx in &problem.order {
        let pool = &pools[region_idx];
        let use_par = should_parallelize_beam(beam.len(), pool.len(), workers);
        parallel_beam_used |= use_par;

        let next_states: Vec<(f64, SearchState, Vec<u16>)> = if use_par {
            // Shared immutable LinkIntern (seeded) — no per-chunk clone.
            // Dominance + select_beam are order-independent / sorted, so no
            // pre-sort of the flat expansion list is required.
            let n_chunks = workers.min(beam.len()).max(1);
            let chunk_size = beam.len().div_ceil(n_chunks).max(1);
            type ExpandRow = (Vec<(f64, SearchState, Vec<u16>)>, u64, u64);
            let rows: Vec<ExpandRow> = beam
                .par_chunks(chunk_size)
                .map(|chunk| {
                    let mut local = Vec::with_capacity(chunk.len().saturating_mul(pool.len()));
                    let mut expanded = 0u64;
                    let mut pruned = 0u64;
                    for state in chunk {
                        let parent_seq = device_seq_of(state, problem, &pools);
                        for &(cand_idx, ref candidate) in pool {
                            expanded += 1;
                            match extend_state(
                                state,
                                region_idx,
                                cand_idx,
                                candidate,
                                problem,
                                &links,
                                &allowed_mask,
                            ) {
                                Some(extended) => {
                                    let mut device_seq = parent_seq.clone();
                                    device_seq.push(candidate.device as u16);
                                    let score = analytic_score(
                                        &extended,
                                        &problem.config,
                                        &problem.capacities,
                                    );
                                    local.push((score, extended, device_seq));
                                }
                                None => pruned += 1,
                            }
                        }
                    }
                    (local, expanded, pruned)
                })
                .collect();
            let mut flat = Vec::new();
            for (mut row, exp, pr) in rows {
                states_expanded += exp;
                states_pruned += pr;
                flat.append(&mut row);
            }
            flat
        } else {
            let mut next_states = Vec::new();
            for state in &beam {
                let parent_seq = device_seq_of(state, problem, &pools);
                for &(cand_idx, ref candidate) in pool {
                    states_expanded += 1;
                    match extend_state(
                        state,
                        region_idx,
                        cand_idx,
                        candidate,
                        problem,
                        &links,
                        &allowed_mask,
                    ) {
                        Some(extended) => {
                            let mut device_seq = parent_seq.clone();
                            device_seq.push(candidate.device as u16);
                            let score =
                                analytic_score(&extended, &problem.config, &problem.capacities);
                            next_states.push((score, extended, device_seq));
                        }
                        None => states_pruned += 1,
                    }
                }
            }
            next_states
        };
        if next_states.is_empty() {
            return (Vec::new(), parallel_beam_used);
        }

        let mut dominant: HashMap<DomKey, (f64, SearchState, Vec<u16>)> = HashMap::new();
        for (score, state, device_seq) in next_states {
            let key = dominance_key(&state, &device_seq);
            match dominant.get(&key) {
                Some((prev_score, prev_state, _))
                    if *prev_score < score
                        || (*prev_score - score).abs() < 1e-18
                            && prev_state.placement_cands <= state.placement_cands => {}
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

    if beam.is_empty() {
        return (Vec::new(), parallel_beam_used);
    }

    // Rank terminal beam; improve the best via local search; keep other distinct terminals.
    beam.sort_by(|a, b| {
        analytic_score(a, &problem.config, &problem.capacities)
            .partial_cmp(&analytic_score(b, &problem.config, &problem.capacities))
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.placement_cands.cmp(&b.placement_cands))
    });

    let (improved_best, local_improvements) = local_search_improve(
        beam[0].clone(),
        &pools,
        problem,
        &links,
        &allowed_mask,
        &consumers,
        &mut states_expanded,
        &mut states_pruned,
    );

    let mut terminals: Vec<SearchState> = Vec::with_capacity(beam.len() + 1);
    terminals.push(improved_best);
    for state in beam.into_iter().skip(1) {
        terminals.push(state);
    }
    terminals.sort_by(|a, b| {
        analytic_score(a, &problem.config, &problem.capacities)
            .partial_cmp(&analytic_score(b, &problem.config, &problem.capacities))
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.placement_cands.cmp(&b.placement_cands))
    });

    let keep = problem.config.resolved_per_subset_finalists();
    let mut out = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for state in terminals {
        let sig = state.placement_cands.clone();
        if !seen.insert(sig) {
            continue;
        }
        out.push(result_from_state(
            &state,
            problem,
            &pools,
            &subset_devices,
            states_expanded,
            states_pruned,
            local_improvements,
        ));
        if out.len() >= keep {
            break;
        }
    }
    (out, parallel_beam_used)
}

/// Plan across all subsets; return diverse top-K finalists.
pub fn plan_placements(problem: &PlanningProblem) -> PlannerOutput {
    let t0 = Instant::now();
    let workers_req = problem.config.planner_workers;
    let workers_available = resolve_workers(workers_req);
    let subset_parallel_gate = should_parallelize_subsets(
        problem.subsets.len(),
        problem.region_count(),
        problem.config.beam_width,
        problem.avg_candidates_per_region(),
        problem.config.allow_parallel_subsets,
        workers_available,
    );
    let beam_may_parallel =
        !subset_parallel_gate && beam_parallelism_possible(problem, workers_available);
    // Subset-level and beam-level Rayon are mutually exclusive: nested pools
    // thrash. Prefer subset parallel when the work estimate says so; otherwise
    // allow intra-subset beam parallel for large single-/few-subset searches.

    // Desired pool size when a parallel path can fire; 1 keeps search serial.
    let desired_pool_threads = if subset_parallel_gate && problem.subsets.len() > 1 {
        workers_available.min(problem.subsets.len()).max(1)
    } else if beam_may_parallel {
        // Cap beam pool: chunked expand only needs modest concurrency.
        // Oversubscribing (e.g. 20 threads on beam=32) loses to serial.
        workers_available.clamp(1, 8)
    } else {
        1
    };
    // Only install a local pool when multi-threaded. Build failure → serial
    // (never fall through to the process-global Rayon pool).
    let local_pool = if desired_pool_threads > 1 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(desired_pool_threads)
            .build()
            .ok()
    } else {
        None
    };
    let pool_threads = if local_pool.is_some() {
        desired_pool_threads
    } else {
        1
    };
    let parallel_subsets = subset_parallel_gate && problem.subsets.len() > 1 && pool_threads > 1;
    let beam_workers = if beam_may_parallel && pool_threads > 1 {
        pool_threads
    } else {
        1
    };

    let (results, parallel_beam_used): (Vec<Vec<SearchResult>>, bool) = {
        let collect = || -> (Vec<Vec<SearchResult>>, bool) {
            let mut beam_par = false;
            let batches: Vec<Vec<SearchResult>> = if parallel_subsets {
                // Subset-level parallel: keep intra-subset beam serial (no nested fanout).
                problem
                    .subsets
                    .par_iter()
                    .map(|subset| {
                        let (r, _) = search_subset_ex(problem, &subset.device_indices, 1);
                        r
                    })
                    .collect()
            } else {
                // Serial subsets; beam may parallelize inside this same pool when sized >1.
                problem
                    .subsets
                    .iter()
                    .map(|subset| {
                        let (r, used) =
                            search_subset_ex(problem, &subset.device_indices, beam_workers);
                        beam_par |= used;
                        r
                    })
                    .collect()
            };
            (batches, beam_par)
        };
        match &local_pool {
            Some(pool) => pool.install(collect),
            None => collect(),
        }
    };

    let mut expanded = 0u64;
    let mut pruned = 0u64;
    let mut local_improvements = 0u32;
    let mut feasible: Vec<SearchResult> = Vec::new();
    for batch in results {
        for r in batch {
            expanded += r.states_expanded;
            pruned += r.states_pruned;
            local_improvements = local_improvements.max(r.local_improvements);
            feasible.push(r);
        }
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

    // First pass: keep several per subset, but cap so one subset cannot monopolize K
    // when other competitive subsets exist. Same-subset alternatives still survive
    // via subset_cap >= per_subset auto (and the uncapped fill pass).
    let per_subset_cap = problem.config.resolved_per_subset_finalists().min(k).max(1);
    for (analytic_rank, result) in feasible.iter().enumerate() {
        if finalists.len() >= k {
            break;
        }
        let sig = placement_signature(&result.placements);
        if !seen_sig.insert(sig.clone()) {
            continue;
        }
        let subset_key = result.subset_devices.clone();
        let count = subset_counts.entry(subset_key).or_insert(0);
        if *count >= per_subset_cap {
            // Defer extras from this subset until the fill pass.
            continue;
        }
        *count += 1;
        let finalist_rank = finalists.len();
        finalists.push(finalist_from_result(
            result,
            analytic_rank,
            finalist_rank,
            sig,
        ));
    }

    // Fill remaining slots without subset cap.
    if finalists.len() < k {
        seen_sig.clear();
        for f in &finalists {
            seen_sig.insert(f.placement_signature.clone());
        }
        for (analytic_rank, result) in feasible.iter().enumerate() {
            if finalists.len() >= k {
                break;
            }
            let sig = placement_signature(&result.placements);
            if !seen_sig.insert(sig.clone()) {
                continue;
            }
            let finalist_rank = finalists.len();
            finalists.push(finalist_from_result(
                result,
                analytic_rank,
                finalist_rank,
                sig,
            ));
        }
    }

    // Effective workers: pool size when a parallel path actually ran, else 1.
    let workers_used = if parallel_subsets || parallel_beam_used {
        pool_threads.max(1)
    } else {
        1
    };

    let deduped = generated.saturating_sub(finalists.len());
    PlannerOutput {
        statistics: PlanStatistics {
            planner_engine: "rust".into(),
            planner_workers_requested: workers_req,
            planner_workers_available: workers_available,
            planner_workers_used: workers_used,
            planner_pool_threads: pool_threads,
            parallel_search_used: parallel_subsets,
            parallel_beam_used,
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
