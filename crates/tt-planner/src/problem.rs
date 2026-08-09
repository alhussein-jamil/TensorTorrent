//! Compact native planning problem (integer-indexed, no Python objects).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tt_runtime::MachineModel;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObjectiveKind {
    Latency,
    Throughput,
    Memory,
    Balanced,
    Weighted,
}

impl ObjectiveKind {
    pub fn parse(s: &str) -> Self {
        match s {
            "throughput" => Self::Throughput,
            "memory" => Self::Memory,
            "balanced" => Self::Balanced,
            "weighted" => Self::Weighted,
            _ => Self::Latency,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PlanningConfig {
    pub objective: ObjectiveKind,
    pub weight_latency: f64,
    pub weight_throughput: f64,
    pub weight_memory: f64,
    pub beam_width: usize,
    pub candidates_per_device: usize,
    pub local_search_iters: usize,
    pub target_inflight_requests: usize,
    pub allow_host_staged_transfers: bool,
    pub vram_budget_bytes: Option<u64>,
    /// 0 = auto, 1 = serial, >1 = cap.
    pub planner_workers: usize,
    /// When false, force serial subset search.
    pub allow_parallel_subsets: bool,
    /// Max distinct finalists returned globally.
    pub finalist_count: usize,
    /// Max distinct terminal plans retained per device subset (`0` = auto).
    pub per_subset_finalists: usize,
}

impl Default for PlanningConfig {
    fn default() -> Self {
        Self {
            objective: ObjectiveKind::Latency,
            weight_latency: 1.0,
            weight_throughput: 0.0,
            weight_memory: 0.0,
            beam_width: 64,
            candidates_per_device: 2,
            local_search_iters: 2,
            target_inflight_requests: 1,
            allow_host_staged_transfers: true,
            vram_budget_bytes: None,
            planner_workers: 0,
            allow_parallel_subsets: true,
            finalist_count: 12,
            per_subset_finalists: 0,
        }
    }
}

impl PlanningConfig {
    /// Distinct terminal plans to keep from one subset before global merge.
    #[must_use]
    pub fn resolved_per_subset_finalists(&self) -> usize {
        if self.per_subset_finalists > 0 {
            return self.per_subset_finalists.max(1);
        }
        // Auto: keep enough alternatives for DES to overturn the analytic winner,
        // but do not let one accelerator-heavy subset consume the whole shortlist
        // before CPU-only / mixed-device baselines reach simulation.
        self.finalist_count.clamp(1, 2)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RegionSpec {
    pub name: String,
    pub depends_on: Vec<usize>,
    pub output_bytes: u64,
    pub state_bytes: u64,
    pub consumer_count: i32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CandidateKernel {
    pub device: usize,
    pub backend_id: String,
    pub kernel_id: String,
    pub dtype: String,
    pub estimated_latency_s: f64,
    pub workspace_bytes: u64,
    pub measured: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SubsetSpec {
    pub device_indices: Vec<usize>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PlacementRecord {
    pub region_id: String,
    pub device: String,
    pub backend_id: String,
    pub dtype: String,
    pub kernel_id: String,
    pub estimated_latency_s: f64,
    pub depends_on: Vec<String>,
    pub measured: bool,
    pub output_bytes: u64,
    pub state_bytes: u64,
    pub workspace_bytes: u64,
}

/// Fully resolved planning input. Strings exist only for result mapping.
#[derive(Clone, Debug)]
pub struct PlanningProblem {
    pub regions: Vec<RegionSpec>,
    /// Topological order as region indices (stable Kahn).
    pub order: Vec<usize>,
    /// Per-region candidate pools (already filtered / ranked).
    pub candidates: Vec<Vec<CandidateKernel>>,
    pub device_names: Vec<String>,
    /// Capacity bytes per device index.
    pub capacities: Vec<u64>,
    /// Memory endpoint name per device (for transfers).
    pub device_memory: Vec<String>,
    /// Edge (producer_region, consumer_region) -> bytes.
    pub edge_bytes: HashMap<(usize, usize), u64>,
    pub subsets: Vec<SubsetSpec>,
    pub machine: MachineModel,
    pub config: PlanningConfig,
}

impl PlanningProblem {
    pub fn region_count(&self) -> usize {
        self.regions.len()
    }

    pub fn avg_candidates_per_region(&self) -> usize {
        if self.candidates.is_empty() {
            return 0;
        }
        let total: usize = self.candidates.iter().map(|c| c.len()).sum();
        total / self.candidates.len()
    }
}

#[cfg(test)]
mod tests {
    use super::PlanningConfig;

    #[test]
    fn auto_finalists_preserve_cross_subset_diversity() {
        let config = PlanningConfig {
            finalist_count: 12,
            ..PlanningConfig::default()
        };
        assert_eq!(config.resolved_per_subset_finalists(), 2);
    }

    #[test]
    fn explicit_per_subset_finalists_remain_honored() {
        let config = PlanningConfig {
            finalist_count: 12,
            per_subset_finalists: 5,
            ..PlanningConfig::default()
        };
        assert_eq!(config.resolved_per_subset_finalists(), 5);
    }
}
