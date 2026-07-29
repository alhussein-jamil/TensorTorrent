//! Minimal machine model for simulation (CPU + virtual accelerators).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MemoryResource {
    pub name: String,
    pub capacity_bytes: u64,
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
    /// resource -> default compute delay scale (1.0 = use instruction predicted_duration)
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
                memory_class: "host".into(),
            },
        );
        Self {
            compute,
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
                name: mem_name,
                capacity_bytes: vram_bytes,
                memory_class: "device".into(),
            },
        );
        self.links.push(TransferLink {
            source: "cpu".into(),
            destination: name.to_owned(),
            bandwidth_bytes_per_s: 12e9,
            latency_s: 1e-5,
        });
        self.links.push(TransferLink {
            source: name.to_owned(),
            destination: "cpu".into(),
            bandwidth_bytes_per_s: 12e9,
            latency_s: 1e-5,
        });
        self
    }

    #[must_use]
    pub fn transfer_time(&self, src: &str, dst: &str, nbytes: u64) -> f64 {
        if src == dst {
            return 0.0;
        }
        for link in &self.links {
            if link.source == src && link.destination == dst {
                let bw = link.bandwidth_bytes_per_s.max(1.0);
                return link.latency_s + (nbytes as f64) / bw;
            }
        }
        // Host memcpy / disk fallback estimate (labelled simulated).
        let bw = if src.contains("disk") || dst.contains("disk") || src == "nvme_or_pack" {
            2e9
        } else {
            20e9
        };
        1e-5 + (nbytes as f64) / bw
    }
}
