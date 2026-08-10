use serde::{Deserialize, Serialize};

/// Every cost must declare provenance.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CostStatus {
    Measured,
    Simulated,
    Estimated,
    Unknown,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RegionCost {
    pub region_id: String,
    pub resource: String,
    pub latency_s: f64,
    pub workspace_bytes: u64,
    pub status: CostStatus,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProfileRecord {
    pub cache_key: String,
    pub costs: Vec<RegionCost>,
    pub transfer_latency_s: f64,
    pub io_latency_s: f64,
    pub status: CostStatus,
    #[serde(default)]
    pub notes: Vec<String>,
}

impl ProfileRecord {
    #[must_use]
    pub fn median_latency(costs: &[RegionCost]) -> Option<f64> {
        if costs.is_empty() {
            return None;
        }
        let mut v: Vec<f64> = costs.iter().map(|c| c.latency_s).collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mid = v.len() / 2;
        Some(if v.len() % 2 == 0 {
            (v[mid - 1] + v[mid]) / 2.0
        } else {
            v[mid]
        })
    }
}
