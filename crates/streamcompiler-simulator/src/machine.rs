//! Minimal machine model for simulation (CPU + virtual accelerators).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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
    pub source: String,
    pub destination: String,
    pub bandwidth_bytes_per_s: f64,
    pub latency_s: f64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct MachineModel {
    pub compute: HashMap<String, f64>,
    /// compute resource id -> primary memory resource id
    #[serde(default)]
    pub memory_affinity: HashMap<String, String>,
    pub memory: HashMap<String, MemoryResource>,
    pub links: Vec<TransferLink>,
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
        }
    }

    #[must_use]
    pub fn with_virtual_accelerator(mut self, name: &str, vram_bytes: u64) -> Self {
        self.compute.insert(name.to_owned(), 1.0);
        let mem_name = format!("{name}_vram");
        self.memory.insert(
            mem_name.clone(),
            MemoryResource {
                name: mem_name.clone(),
                capacity_bytes: vram_bytes,
                allocatable_bytes: 0,
                memory_class: "device".into(),
            },
        );
        self.memory_affinity.insert(name.to_owned(), mem_name.clone());
        self.links.push(TransferLink {
            source: "host_ram".into(),
            destination: mem_name.clone(),
            bandwidth_bytes_per_s: 12e9,
            latency_s: 1e-5,
        });
        self.links.push(TransferLink {
            source: mem_name,
            destination: "host_ram".into(),
            bandwidth_bytes_per_s: 12e9,
            latency_s: 1e-5,
        });
        self
    }

    fn resolve_endpoint<'a>(&'a self, endpoint: &'a str) -> &'a str {
        if self.memory.contains_key(endpoint) {
            return endpoint;
        }
        if let Some(aff) = self.memory_affinity.get(endpoint) {
            return aff.as_str();
        }
        endpoint
    }

    #[must_use]
    pub fn transfer_time(&self, src: &str, dst: &str, nbytes: u64) -> f64 {
        let src_m = self.resolve_endpoint(src);
        let dst_m = self.resolve_endpoint(dst);
        if src_m == dst_m {
            return 0.0;
        }
        for link in &self.links {
            let ls = self.resolve_endpoint(&link.source);
            let ld = self.resolve_endpoint(&link.destination);
            if (ls == src_m && ld == dst_m)
                || (link.source == src && link.destination == dst)
                || (link.source == src_m && link.destination == dst_m)
            {
                let bw = link.bandwidth_bytes_per_s.max(1.0);
                return link.latency_s + (nbytes as f64) / bw;
            }
        }
        // Also try raw compute ids against link endpoints.
        for link in &self.links {
            if link.source == src && link.destination == dst {
                let bw = link.bandwidth_bytes_per_s.max(1.0);
                return link.latency_s + (nbytes as f64) / bw;
            }
        }
        let bw = if src.contains("disk") || dst.contains("disk") || src == "nvme_or_pack" {
            2e9
        } else {
            20e9
        };
        1e-5 + (nbytes as f64) / bw
    }
}
