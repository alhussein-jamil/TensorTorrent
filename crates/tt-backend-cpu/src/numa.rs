//! NUMA topology discovery for the CPU backend.
//!
//! Discovery is always on. Thread/process binding via `numactl`/cgroups is
//! deferred until multi-socket hosts need it (`NumaTopology.sockets > 1`).

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NumaNode {
    pub node_id: u32,
    pub cpu_list: Vec<u32>,
    pub memory_bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NumaTopology {
    pub nodes: Vec<NumaNode>,
    pub sockets: u32,
}

impl NumaTopology {
    #[must_use]
    pub fn total_logical_cpus(&self) -> usize {
        self.nodes.iter().map(|n| n.cpu_list.len()).sum()
    }
}

/// Discover NUMA nodes from sysfs. Falls back to a single domain covering all CPUs.
#[must_use]
pub fn discover_numa_topology() -> NumaTopology {
    let base = Path::new("/sys/devices/system/node");
    let mut nodes = Vec::new();
    if base.is_dir() {
        if let Ok(entries) = fs::read_dir(base) {
            let mut names: Vec<_> = entries
                .flatten()
                .map(|e| e.file_name())
                .filter(|n| {
                    let s = n.to_string_lossy();
                    s.starts_with("node") && s[4..].chars().all(|c| c.is_ascii_digit())
                })
                .collect();
            names.sort();
            for name in names {
                let s = name.to_string_lossy();
                let id: u32 = match s[4..].parse() {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                let node_path = base.join(&*s);
                let cpu_list = read_cpulist(&node_path.join("cpulist"));
                let memory_bytes = read_node_meminfo(&node_path.join("meminfo"));
                nodes.push(NumaNode {
                    node_id: id,
                    cpu_list,
                    memory_bytes,
                });
            }
        }
    }
    if nodes.is_empty() {
        let cpus = online_cpu_count();
        let mem = host_memory_bytes();
        nodes.push(NumaNode {
            node_id: 0,
            cpu_list: (0..cpus).collect(),
            memory_bytes: mem,
        });
    }
    let sockets = crate::host_sys::socket_count();
    NumaTopology { nodes, sockets }
}

fn read_cpulist(path: &Path) -> Vec<u32> {
    let Ok(text) = fs::read_to_string(path) else {
        return (0..online_cpu_count()).collect();
    };
    parse_cpu_list(text.trim())
}

fn parse_cpu_list(text: &str) -> Vec<u32> {
    let mut out = Vec::new();
    if text.is_empty() {
        return out;
    }
    for part in text.split(',') {
        if let Some((a, b)) = part.split_once('-') {
            if let (Ok(start), Ok(end)) = (a.parse::<u32>(), b.parse::<u32>()) {
                out.extend(start..=end);
            }
        } else if let Ok(v) = part.parse::<u32>() {
            out.push(v);
        }
    }
    out
}

fn read_node_meminfo(path: &Path) -> u64 {
    let Ok(text) = fs::read_to_string(path) else {
        return host_memory_bytes();
    };
    for line in text.lines() {
        // Node 0 MemTotal:       12345 kB
        if let Some(idx) = line.find("MemTotal:") {
            let rest = &line[idx + "MemTotal:".len()..];
            let kb: u64 = rest
                .split_whitespace()
                .next()
                .and_then(|t| t.parse().ok())
                .unwrap_or(0);
            if kb > 0 {
                return kb.saturating_mul(1024);
            }
        }
    }
    host_memory_bytes()
}

fn host_memory_bytes() -> u64 {
    crate::host_sys::memory_total_bytes()
        .or_else(crate::host_sys::memory_available_bytes)
        .unwrap_or(0)
}

fn online_cpu_count() -> u32 {
    crate::host_sys::online_cpu_count() as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_ranges() {
        assert_eq!(parse_cpu_list("0-3,8"), vec![0, 1, 2, 3, 8]);
    }

    #[test]
    fn discover_nonempty() {
        let topo = discover_numa_topology();
        assert!(!topo.nodes.is_empty());
        assert!(topo.total_logical_cpus() >= 1);
        let mem: u64 = topo.nodes.iter().map(|n| n.memory_bytes).sum();
        assert!(
            mem >= 128 * 1024 * 1024,
            "NUMA fallback must report real host RAM; got {mem}"
        );
    }
}
