//! Analytic pruning scores and finalist comparable scores.
//!
//! Intra-beam analytic scores and cross-subset comparable scores intentionally
//! differ. Final DES ranking uses a separate authoritative score.

use crate::problem::{ObjectiveKind, PlanningConfig};
use crate::search::SearchState;

#[must_use]
pub fn memory_pressure(peak_bytes: &[u64], capacities: &[u64]) -> f64 {
    let mut pressure = 0.0;
    for (i, &peak) in peak_bytes.iter().enumerate() {
        let capacity = capacities.get(i).copied().unwrap_or(0);
        if capacity > 0 {
            pressure += peak as f64 / capacity as f64;
        } else if peak > 0 {
            pressure += 1.0;
        }
    }
    pressure
}

/// Beam / local-search score (lower better). Mirrors Python `_state_score`.
#[must_use]
pub fn analytic_score(state: &SearchState, config: &PlanningConfig, capacities: &[u64]) -> f64 {
    let latency = state.makespan_s();
    let cycle = state.initiation_interval_s();
    let peak = state.peak_bytes.iter().map(|&b| b as f64).sum::<f64>();
    let pressure = memory_pressure(&state.peak_bytes, capacities);

    match config.objective {
        ObjectiveKind::Latency => latency + 1e-6 * cycle + 1e-9 * pressure,
        ObjectiveKind::Throughput => cycle + 1e-3 * latency + 1e-9 * pressure,
        ObjectiveKind::Memory => peak + 1e-3 * pressure + 1e-9 * latency,
        ObjectiveKind::Balanced => latency + cycle + 0.05 * pressure,
        ObjectiveKind::Weighted => {
            config.weight_latency * latency
                + config.weight_throughput * cycle
                + config.weight_memory * pressure
        }
    }
}

/// Cross-subset / finalist shortlist score (lower better).
/// Mirrors Python `_comparable_search_score`.
#[must_use]
pub fn comparable_finalist_score(
    latency_s: f64,
    throughput_per_s: f64,
    peak_bytes: &[u64],
    capacities: &[u64],
    config: &PlanningConfig,
) -> f64 {
    let peak_total: u64 = peak_bytes.iter().sum();
    let pressure = memory_pressure(peak_bytes, capacities);
    match config.objective {
        ObjectiveKind::Throughput => 1.0 / throughput_per_s.max(1e-12),
        ObjectiveKind::Memory => peak_total as f64 + 1e-9 * latency_s,
        ObjectiveKind::Balanced => latency_s + 1.0 / throughput_per_s.max(1e-12) + 0.05 * pressure,
        ObjectiveKind::Weighted => {
            config.weight_latency * latency_s
                + config.weight_throughput * (1.0 / throughput_per_s.max(1e-12))
                + config.weight_memory * pressure
        }
        ObjectiveKind::Latency => latency_s,
    }
}
