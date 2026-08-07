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
    if !allow_parallel || workers <= 1 || subset_count < 3 || region_count < 2 {
        return false;
    }
    // Rough expansion count: subsets * regions * beam * cand.
    let work = subset_count
        .saturating_mul(region_count)
        .saturating_mul(beam_width.max(1))
        .saturating_mul(candidates_per_region_avg.max(1));
    work >= 2_000
}
