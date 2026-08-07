//! Native heterogeneous placement planner.
//!
//! Two-stage optimizer:
//! 1. Fast incremental beam search over device subsets (Rayon-parallel).
//! 2. Caller constructs real schedules and ranks finalists with DES.
//!
//! Hot loops never touch Python. Convert once at the PyO3 boundary.

mod problem;
mod score;
mod search;

#[cfg(test)]
mod oracle;
#[cfg(test)]
#[path = "tests_planner.rs"]
mod tests_planner;

pub use problem::{
    CandidateKernel, ObjectiveKind, PlacementRecord, PlanningConfig, PlanningProblem, RegionSpec,
    SubsetSpec,
};
pub use score::{analytic_score, comparable_finalist_score, memory_pressure};
pub use search::{
    plan_placements, search_subset, FinalistPlan, PlanStatistics, PlannerOutput, SearchResult,
};

/// Whether a single subset's beam expansion is large enough to parallelize.
#[must_use]
pub fn should_parallelize_beam(beam_len: usize, pool_len: usize, workers: usize) -> bool {
    if workers <= 1 {
        return false;
    }
    // Bench (planner_native_bench): Mutex-per-extend hurt; clone-per-parent needs
    // enough fanout to beat serial. Threshold tuned so ~16-region / small-pool
    // graphs stay serial; large beams gain.
    beam_len.saturating_mul(pool_len.max(1)) >= 512
}

/// Upper-bound check: could any beam step in this problem hit the parallel gate?
#[must_use]
pub fn beam_parallelism_possible(problem: &PlanningProblem, workers: usize) -> bool {
    if workers <= 1 {
        return false;
    }
    let per_device = problem.config.candidates_per_device.max(1);
    let n_dev = problem.device_names.len().max(1);
    let max_pool = problem
        .candidates
        .iter()
        .map(|pool| pool.len().min(per_device.saturating_mul(n_dev)))
        .max()
        .unwrap_or(0);
    should_parallelize_beam(problem.config.beam_width.max(1), max_pool.max(1), workers)
}

/// Resolve worker count: `0` → available parallelism, `1` → serial, else capped.
#[must_use]
pub fn resolve_workers(requested: usize) -> usize {
    match requested {
        0 => std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .max(1),
        n => n.max(1),
    }
}

/// Cheap work estimate deciding whether subset parallelism pays off.
#[must_use]
pub fn should_parallelize_subsets(
    subset_count: usize,
    region_count: usize,
    beam_width: usize,
    candidates_per_region_avg: usize,
    allow_parallel: bool,
    workers: usize,
) -> bool {
    if !allow_parallel || workers <= 1 || subset_count < 4 || region_count < 4 {
        return false;
    }
    // Rough expansion count: subsets * regions * beam * cand.
    // Threshold tuned so tiny/medium graphs stay serial (thread setup dominates).
    let work = subset_count
        .saturating_mul(region_count)
        .saturating_mul(beam_width.max(1))
        .saturating_mul(candidates_per_region_avg.max(1));
    work >= 8_000
}
