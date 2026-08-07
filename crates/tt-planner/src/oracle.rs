//! Exhaustive enumeration oracle for tiny planning problems (tests / differentials).

use crate::problem::{CandidateKernel, PlanningProblem};
use crate::score::analytic_score;
use crate::search::{
    extend_state_public, filter_pool_public, replay_assignment_public, seed_link_intern,
    LinkIntern, SearchState,
};
use std::cmp::Ordering;

fn allowed_mask(problem: &PlanningProblem, subset_device_indices: &[usize]) -> Vec<bool> {
    let mut allowed = vec![false; problem.device_names.len()];
    for &d in subset_device_indices {
        if d < allowed.len() {
            allowed[d] = true;
        }
    }
    allowed
}

/// Brute-force best analytic score over all candidate assignments on one subset.
///
/// Only for tiny problems. Returns `None` if empty or node budget exceeded.
pub fn exhaustive_best(
    problem: &PlanningProblem,
    subset_device_indices: &[usize],
    max_nodes: usize,
) -> Option<(f64, Vec<u16>)> {
    let allowed = allowed_mask(problem, subset_device_indices);
    let per_device = problem.config.candidates_per_device.max(1);
    let pools: Vec<Vec<(u16, CandidateKernel)>> = problem
        .candidates
        .iter()
        .map(|pool| filter_pool_public(pool, &allowed, per_device))
        .collect();
    for &r in &problem.order {
        if pools[r].is_empty() {
            return None;
        }
    }

    let consumers: Vec<i32> = problem.regions.iter().map(|r| r.consumer_count).collect();
    let mut best_score = f64::INFINITY;
    let mut best_cands: Vec<u16> = Vec::new();
    let mut visited = 0usize;
    let mut links = LinkIntern::new();
    seed_link_intern(problem, &mut links);

    struct Scratch<'a> {
        problem: &'a PlanningProblem,
        pools: &'a [Vec<(u16, CandidateKernel)>],
        allowed: &'a [bool],
        links: &'a LinkIntern,
        max_nodes: usize,
        visited: &'a mut usize,
        best_score: &'a mut f64,
        best_cands: &'a mut Vec<u16>,
        path: &'a mut Vec<u16>,
    }

    fn rec(step: usize, state: &SearchState, scratch: &mut Scratch<'_>) {
        if *scratch.visited >= scratch.max_nodes {
            return;
        }
        if step == scratch.problem.order.len() {
            *scratch.visited += 1;
            let score = analytic_score(state, &scratch.problem.config, &scratch.problem.capacities);
            if scratch.best_cands.is_empty()
                || score < *scratch.best_score
                || ((score - *scratch.best_score).abs() < 1e-15
                    && scratch.path.as_slice() < scratch.best_cands.as_slice())
            {
                *scratch.best_score = score;
                *scratch.best_cands = scratch.path.clone();
            }
            return;
        }
        let ridx = scratch.problem.order[step];
        for &(ci, ref c) in &scratch.pools[ridx] {
            if *scratch.visited >= scratch.max_nodes {
                return;
            }
            *scratch.visited += 1;
            if let Some(next) = extend_state_public(
                state,
                ridx,
                ci,
                c,
                scratch.problem,
                scratch.links,
                scratch.allowed,
            ) {
                scratch.path.push(ci);
                rec(step + 1, &next, scratch);
                scratch.path.pop();
            }
        }
    }

    let root = SearchState::new(
        problem.regions.len(),
        problem.device_names.len(),
        &consumers,
    );
    let mut path = Vec::new();
    rec(
        0,
        &root,
        &mut Scratch {
            problem,
            pools: &pools,
            allowed: &allowed,
            links: &links,
            max_nodes,
            visited: &mut visited,
            best_score: &mut best_score,
            best_cands: &mut best_cands,
            path: &mut path,
        },
    );
    if best_cands.is_empty() {
        None
    } else {
        Some((best_score, best_cands))
    }
}

/// Compare prefix-replay local mutation against full replay for one alternate.
pub fn replay_scores_match(
    problem: &PlanningProblem,
    subset_device_indices: &[usize],
    assignment: &[(u16, CandidateKernel)],
    mutate_index: usize,
    alternate: &(u16, CandidateKernel),
) -> bool {
    let allowed = allowed_mask(problem, subset_device_indices);
    let consumers: Vec<i32> = problem.regions.iter().map(|r| r.consumer_count).collect();
    let mut trial = assignment.to_vec();
    trial[mutate_index] = alternate.clone();
    let n_dev = problem.device_names.len();

    let mut links_full = LinkIntern::new();
    seed_link_intern(problem, &mut links_full);
    let full = {
        let mut state = SearchState::new(problem.regions.len(), n_dev, &consumers);
        let mut ok = true;
        for (step, &ridx) in problem.order.iter().enumerate() {
            let (ci, ref c) = trial[step];
            match extend_state_public(&state, ridx, ci, c, problem, &links_full, &allowed) {
                Some(next) => state = next,
                None => {
                    ok = false;
                    break;
                }
            }
        }
        if ok {
            Some(state)
        } else {
            None
        }
    };

    let mut links_pref = LinkIntern::new();
    seed_link_intern(problem, &mut links_pref);
    let mut prefixes = vec![SearchState::new(problem.regions.len(), n_dev, &consumers)];
    {
        let mut cursor = prefixes[0].clone();
        for (step, &ridx) in problem.order.iter().enumerate().take(mutate_index) {
            let (ci, ref c) = assignment[step];
            match extend_state_public(&cursor, ridx, ci, c, problem, &links_pref, &allowed) {
                Some(next) => {
                    cursor = next;
                    prefixes.push(cursor.clone());
                }
                None => return full.is_none(),
            }
        }
    }
    let prefix = prefixes.get(mutate_index);
    let mut links_pref2 = LinkIntern::new();
    seed_link_intern(problem, &mut links_pref2);
    let prefixed = replay_assignment_public(
        &trial,
        problem,
        &links_pref2,
        &allowed,
        &consumers,
        mutate_index,
        prefix,
    );

    match (full, prefixed) {
        (None, None) => true,
        (Some(a), Some(b)) => {
            let sa = analytic_score(&a, &problem.config, &problem.capacities);
            let sb = analytic_score(&b, &problem.config, &problem.capacities);
            (sa - sb).abs() < 1e-12
                && a.placement_cands.cmp(&b.placement_cands) == Ordering::Equal
                && (a.makespan_s() - b.makespan_s()).abs() < 1e-12
        }
        _ => false,
    }
}
