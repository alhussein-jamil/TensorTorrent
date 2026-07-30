//! NUMA topology discovery for the CPU backend.

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
    let sockets = read_socket_count().max(1);
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
    let Ok(text) = fs::read_to_string("/proc/meminfo") else {
        return 0;
    };
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            let kb: u64 = rest
                .split_whitespace()
                .next()
                .and_then(|t| t.parse().ok())
                .unwrap_or(0);
            return kb.saturating_mul(1024);
        }
    }
    0
}

fn online_cpu_count() -> u32 {
    fs::read_to_string("/proc/cpuinfo")
        .ok()
        .map(|t| t.lines().filter(|l| l.starts_with("processor")).count() as u32)
        .unwrap_or(1)
        .max(1)
}

fn read_socket_count() -> u32 {
    let Ok(text) = fs::read_to_string("/proc/cpuinfo") else {
        return 1;
    };
    let mut ids: std::collections::BTreeSet<u32> = std::collections::BTreeSet::new();
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("physical id") {
            if let Some(v) = rest.split(':').nth(1).and_then(|s| s.trim().parse().ok()) {
                ids.insert(v);
            }
        }
    }
    ids.len().max(1) as u32
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
    }
}
