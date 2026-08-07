//! Machine model for planner search and discrete-event simulation.
//!
//! Transfer coefficients are authoritative here: measured values when present,
//! otherwise explicit link-class priors. Both the native planner and DES call
//! [`MachineModel::transfer_time`] so analytic search and schedule simulation
//! share one cost model.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Prior `(latency_s, bandwidth_bytes_per_s)` when a link lacks measured coeffs.
pub fn link_class_prior(link_class: &str) -> (f64, f64) {
    match link_class {
        "cpu_local" => (0.4e-6, 80e9),
        "shared_memory" => (0.7e-6, 120e9),
        "numa_interconnect" => (1.5e-6, 35e9),
        "nvlink" => (2.0e-6, 100e9),
        "infinity_fabric" => (2.5e-6, 80e9),
        "cxl" => (3.0e-6, 40e9),
        "pcie" => (8.0e-6, 12e9),
        "host_staged" => (16.0e-6, 6e9),
        "storage" => (35.0e-6, 2.5e9),
        "network" => (20.0e-6, 10e9),
        _ => (15.0e-6, 8e9), // unknown
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MemoryResource {
    pub name: String,
    pub capacity_bytes: u64,
    /// Soft pressure threshold; 0 means use capacity_bytes.
    #[serde(default)]
    pub allocatable_bytes: u64,
    pub memory_class: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TransferLink {
    #[serde(default)]
    pub id: String,
    pub source: String,
    pub destination: String,
    pub bandwidth_bytes_per_s: f64,
    pub latency_s: f64,
    #[serde(default = "default_link_class")]
    pub link_class: String,
    #[serde(default = "default_contention")]
    pub contention_factor: f64,
    #[serde(default)]
    pub measured: bool,
    #[serde(default)]
    pub bidirectional: bool,
    #[serde(default)]
    pub peer_to_peer: bool,
}

fn default_link_class() -> String {
    "unknown".into()
}

fn default_contention() -> f64 {
    1.0
}

impl TransferLink {
    /// Resolve missing latency/bandwidth from link-class priors.
    ///
    /// When either coefficient is non-positive, fill from priors and clear
    /// `measured` so diagnostics still distinguish real vs prior paths.
    pub fn resolve_priors(&mut self) {
        let (prior_lat, prior_bw) = link_class_prior(&self.link_class);
        let mut used_prior = false;
        if !(self.latency_s.is_finite() && self.latency_s >= 0.0) {
            self.latency_s = prior_lat;
            used_prior = true;
        }
        if !(self.bandwidth_bytes_per_s.is_finite() && self.bandwidth_bytes_per_s > 0.0) {
            self.bandwidth_bytes_per_s = prior_bw;
            used_prior = true;
        }
        if used_prior {
            self.measured = false;
        }
        if !(self.contention_factor.is_finite()) || self.contention_factor < 1.0 {
            self.contention_factor = 1.0;
        }
        if self.id.is_empty() {
            self.id = format!("{}->{}", self.source, self.destination);
        }
    }

    #[must_use]
    pub fn duration(&self, nbytes: u64) -> f64 {
        let bw = self.bandwidth_bytes_per_s.max(1.0);
        let contention = self.contention_factor.max(1.0);
        (self.latency_s + (nbytes as f64) / bw) * contention
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MachineModel {
    pub compute: HashMap<String, f64>,
    /// compute resource id -> primary memory resource id
    #[serde(default)]
    pub memory_affinity: HashMap<String, String>,
    pub memory: HashMap<String, MemoryResource>,
    pub links: Vec<TransferLink>,
    /// When true, missing links use the host-staged prior (planner policy).
    #[serde(default = "default_true")]
    pub allow_host_staged_transfers: bool,
}

fn default_true() -> bool {
    true
}

impl Default for MachineModel {
    fn default() -> Self {
        Self {
            compute: HashMap::new(),
            memory_affinity: HashMap::new(),
            memory: HashMap::new(),
            links: Vec::new(),
            allow_host_staged_transfers: true,
        }
    }
}

/// Outcome of a transfer lookup used by planner search.
#[derive(Clone, Debug)]
pub struct TransferEstimate {
    pub duration_s: f64,
    pub resource: String,
    pub measured: bool,
    pub host_staged: bool,
    pub contention_factor: f64,
}

impl MachineModel {
    #[must_use]
    pub fn cpu_only() -> Self {
        let mut compute = HashMap::new();
        compute.insert("cpu".into(), 1.0);
        let mut memory = HashMap::new();
        memory.insert(
            "host_ram".into(),
            MemoryResource {
                name: "host_ram".into(),
                capacity_bytes: 64 * 1024 * 1024 * 1024,
                allocatable_bytes: 0,
                memory_class: "host".into(),
            },
        );
        let mut memory_affinity = HashMap::new();
        memory_affinity.insert("cpu".into(), "host_ram".into());
        Self {
            compute,
            memory_affinity,
            memory,
            links: vec![],
            allow_host_staged_transfers: true,
        }
    }

    /// Resolve priors on every link once after construction / Python conversion.
    pub fn resolve_all_link_priors(&mut self) {
        for link in &mut self.links {
            link.resolve_priors();
        }
    }

    pub fn resolve_endpoint<'a>(&'a self, endpoint: &'a str) -> &'a str {
        if self.memory.contains_key(endpoint) {
            return endpoint;
        }
        if let Some(aff) = self.memory_affinity.get(endpoint) {
            return aff.as_str();
        }
        endpoint
    }

    fn find_link(&self, src_m: &str, dst_m: &str) -> Option<&TransferLink> {
        for link in &self.links {
            let ls = self.resolve_endpoint(&link.source);
            let ld = self.resolve_endpoint(&link.destination);
            let forward =
                (ls == src_m && ld == dst_m) || (link.source == src_m && link.destination == dst_m);
            let reverse = link.bidirectional
                && ((ls == dst_m && ld == src_m)
                    || (link.source == dst_m && link.destination == src_m));
            if forward || reverse {
                return Some(link);
            }
        }
        None
    }

    /// Duration of a transfer; shared by planner and DES.
    #[must_use]
    pub fn transfer_time(&self, src: &str, dst: &str, nbytes: u64) -> f64 {
        self.estimate_transfer(src, dst, nbytes)
            .map(|e| e.duration_s)
            .unwrap_or(0.0)
    }

    /// Full transfer estimate including measured / host-staged flags.
    ///
    /// Returns `None` only when host-staged is disallowed and no link exists.
    #[must_use]
    pub fn estimate_transfer(&self, src: &str, dst: &str, nbytes: u64) -> Option<TransferEstimate> {
        let src_m = self.resolve_endpoint(src);
        let dst_m = self.resolve_endpoint(dst);
        if src_m == dst_m || nbytes == 0 {
            return Some(TransferEstimate {
                duration_s: 0.0,
                resource: format!("local::{src}"),
                measured: true,
                host_staged: false,
                contention_factor: 1.0,
            });
        }

        if let Some(link) = self.find_link(src_m, dst_m) {
            let host_staged = link.link_class == "host_staged";
            if host_staged && !self.allow_host_staged_transfers {
                return None;
            }
            return Some(TransferEstimate {
                duration_s: link.duration(nbytes),
                resource: if link.id.is_empty() {
                    format!("{}->{}", link.source, link.destination)
                } else {
                    link.id.clone()
                },
                measured: link.measured,
                host_staged,
                contention_factor: link.contention_factor.max(1.0),
            });
        }

        // Storage / disk paths keep a dedicated prior (DES Prefetch/Load).
        if src.contains("disk")
            || dst.contains("disk")
            || src == "nvme_or_pack"
            || dst == "nvme_or_pack"
            || src_m.contains("disk")
            || dst_m.contains("disk")
        {
            let (lat, bw) = link_class_prior("storage");
            return Some(TransferEstimate {
                duration_s: lat + (nbytes as f64) / bw.max(1.0),
                resource: format!("storage::{src_m}->{dst_m}"),
                measured: false,
                host_staged: false,
                contention_factor: 1.0,
            });
        }

        if !self.allow_host_staged_transfers {
            return None;
        }

        // Last-resort host-staged prior — same coefficients as the planner.
        let (lat, bw) = link_class_prior("host_staged");
        Some(TransferEstimate {
            duration_s: lat + (nbytes as f64) / bw.max(1.0),
            resource: format!("host_staged::{src_m}->{dst_m}"),
            measured: false,
            host_staged: true,
            contention_factor: 1.0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contention_scales_duration() {
        let mut m = MachineModel::cpu_only();
        m.memory.insert(
            "vram0".into(),
            MemoryResource {
                name: "vram0".into(),
                capacity_bytes: 1 << 30,
                allocatable_bytes: 1 << 30,
                memory_class: "device_vram".into(),
            },
        );
        m.links.push(TransferLink {
            id: "l0".into(),
            source: "host_ram".into(),
            destination: "vram0".into(),
            bandwidth_bytes_per_s: 10e9,
            latency_s: 1e-5,
            link_class: "pcie".into(),
            contention_factor: 2.0,
            measured: true,
            bidirectional: false,
            peer_to_peer: false,
        });
        let base = 1e-5 + 1e9 / 10e9;
        let t = m.transfer_time("host_ram", "vram0", 1_000_000_000);
        assert!((t - base * 2.0).abs() < 1e-12);
    }

    #[test]
    fn bidirectional_match() {
        let mut m = MachineModel::cpu_only();
        m.memory.insert(
            "vram0".into(),
            MemoryResource {
                name: "vram0".into(),
                capacity_bytes: 1 << 30,
                allocatable_bytes: 1 << 30,
                memory_class: "device_vram".into(),
            },
        );
        m.links.push(TransferLink {
            id: "l0".into(),
            source: "host_ram".into(),
            destination: "vram0".into(),
            bandwidth_bytes_per_s: 12e9,
            latency_s: 8e-6,
            link_class: "pcie".into(),
            contention_factor: 1.0,
            measured: true,
            bidirectional: true,
            peer_to_peer: false,
        });
        assert!(m.estimate_transfer("vram0", "host_ram", 100).is_some());
    }

    #[test]
    fn host_staged_disallowed() {
        let mut m = MachineModel::cpu_only();
        m.allow_host_staged_transfers = false;
        m.memory.insert(
            "vram0".into(),
            MemoryResource {
                name: "vram0".into(),
                capacity_bytes: 1 << 30,
                allocatable_bytes: 1 << 30,
                memory_class: "device_vram".into(),
            },
        );
        assert!(m.estimate_transfer("host_ram", "vram0", 100).is_none());
    }
}
