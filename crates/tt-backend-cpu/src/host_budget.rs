//! Effective host resource budgets: what this **process** may use, not what
//! the machine physically has.
//!
//! Every number the backend sizes itself from must come through here.
//! Precedence per resource:
//!   1. explicit caller override (wired from the Python resolver)
//!   2. cgroup v2, then cgroup v1 (container limits minus current usage)
//!   3. live OS availability (`MemAvailable`, scheduler affinity mask)
//!   4. machine totals — last resort only
//!
//! A reserve floor is always withheld from the memory budget so the host
//! (desktop session, other tenants) keeps working memory even when the
//! budget is fully consumed.

use std::fs;
use std::path::Path;

const KIB: u64 = 1024;
const MIB: u64 = 1024 * KIB;
const GIB: u64 = 1024 * MIB;
/// cgroup v1 reports "unlimited" as a page-rounded i64::MAX.
const CGROUP_V1_UNLIMITED: u64 = 0x7FFF_FFFF_FFFF_F000;

/// Resolved budget with provenance, so diagnostics can explain every number.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HostBudget {
    /// Bytes of host memory this process should treat as its ceiling.
    pub memory_bytes: u64,
    /// Where the memory number came from (e.g. "cgroup_v2", "os_available").
    pub memory_source: &'static str,
    /// Bytes withheld from the raw figure as host headroom.
    pub memory_reserved_bytes: u64,
    /// Logical CPUs this process should size thread pools from.
    pub cpu_count: usize,
    /// Where the CPU number came from (e.g. "affinity", "cgroup_v2_quota").
    pub cpu_source: &'static str,
}

/// Resolve the effective budget from the live system. Never panics; every
/// probe degrades to the next precedence level on failure.
#[must_use]
pub fn effective_host_budget() -> HostBudget {
    let (raw_mem, mem_source) = resolve_memory();
    let reserve = memory_reserve_bytes(raw_mem);
    let memory_bytes = raw_mem.saturating_sub(reserve).max(128 * MIB);
    let (cpu_count, cpu_source) = resolve_cpus();
    HostBudget {
        memory_bytes,
        memory_source: mem_source,
        memory_reserved_bytes: reserve,
        cpu_count,
        cpu_source,
    }
}

/// Headroom withheld for the rest of the host: 5% of the raw budget,
/// clamped to [256 MiB, 2 GiB].
#[must_use]
pub fn memory_reserve_bytes(raw_budget: u64) -> u64 {
    (raw_budget / 20).clamp(256 * MIB, 2 * GIB)
}

fn resolve_memory() -> (u64, &'static str) {
    let cgroup = cgroup_memory_available(Path::new("/sys/fs/cgroup"));
    let os_avail = crate::host_sys::memory_available_bytes();
    match (cgroup, os_avail) {
        (Some((cg, src)), Some(avail)) => {
            if cg <= avail {
                (cg, src)
            } else {
                (avail, "os_available")
            }
        }
        (Some((cg, src)), None) => (cg, src),
        (None, Some(avail)) => (avail, "os_available"),
        (None, None) => (
            crate::host_sys::memory_total_bytes().unwrap_or(0),
            "total_fallback",
        ),
    }
}

/// Container memory still available to this cgroup: limit minus current use.
fn cgroup_memory_available(root: &Path) -> Option<(u64, &'static str)> {
    // v2 unified hierarchy.
    if let Some(limit) = read_cgroup_v2_memory_limit(root) {
        let current = read_u64_file(&root.join("memory.current")).unwrap_or(0);
        return Some((limit.saturating_sub(current), "cgroup_v2"));
    }
    // v1 memory controller.
    let v1 = root.join("memory");
    if let Some(limit) = read_u64_file(&v1.join("memory.limit_in_bytes")) {
        if limit < CGROUP_V1_UNLIMITED {
            let current = read_u64_file(&v1.join("memory.usage_in_bytes")).unwrap_or(0);
            return Some((limit.saturating_sub(current), "cgroup_v1"));
        }
    }
    None
}

fn read_cgroup_v2_memory_limit(root: &Path) -> Option<u64> {
    let max = read_limit_file(&root.join("memory.max"));
    let high = read_limit_file(&root.join("memory.high"));
    match (max, high) {
        (Some(m), Some(h)) => Some(m.min(h)),
        (Some(m), None) => Some(m),
        (None, Some(h)) => Some(h),
        (None, None) => None,
    }
}

/// Reads a cgroup v2 limit file where "max" means unlimited (returns None).
fn read_limit_file(path: &Path) -> Option<u64> {
    let text = fs::read_to_string(path).ok()?;
    parse_limit_text(&text)
}

fn parse_limit_text(text: &str) -> Option<u64> {
    let t = text.trim();
    if t.is_empty() || t == "max" {
        return None;
    }
    t.parse().ok()
}

fn read_u64_file(path: &Path) -> Option<u64> {
    fs::read_to_string(path).ok()?.trim().parse().ok()
}

#[cfg(test)]
fn parse_meminfo_field(text: &str, field: &str) -> Option<u64> {
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix(field) {
            let kb: u64 = rest.split_whitespace().next()?.parse().ok()?;
            return Some(kb.saturating_mul(KIB));
        }
    }
    None
}

fn resolve_cpus() -> (usize, &'static str) {
    let mut best = (crate::host_sys::online_cpu_count(), "online_count");
    if let Some(aff) = affinity_cpu_count(Path::new("/proc/self/status")) {
        if aff < best.0 {
            best = (aff, "affinity");
        }
    }
    if let Some(quota) = cgroup_cpu_quota(Path::new("/sys/fs/cgroup")) {
        if quota < best.0 {
            best = (quota, "cgroup_quota");
        }
    }
    (best.0.max(1), best.1)
}

/// Count of CPUs in the scheduler affinity mask (`taskset`, cpusets).
fn affinity_cpu_count(status: &Path) -> Option<usize> {
    let text = fs::read_to_string(status).ok()?;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("Cpus_allowed_list:") {
            let n = parse_cpu_list_count(rest.trim());
            if n > 0 {
                return Some(n);
            }
        }
    }
    None
}

fn parse_cpu_list_count(text: &str) -> usize {
    let mut count = 0usize;
    for part in text.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        if let Some((a, b)) = part.split_once('-') {
            if let (Ok(start), Ok(end)) = (a.parse::<u64>(), b.parse::<u64>()) {
                if end >= start {
                    count += (end - start + 1) as usize;
                }
            }
        } else if part.parse::<u64>().is_ok() {
            count += 1;
        }
    }
    count
}

/// Whole CPUs implied by the cgroup CPU quota (rounded up), if limited.
fn cgroup_cpu_quota(root: &Path) -> Option<usize> {
    // v2: "quota period" or "max period".
    if let Ok(text) = fs::read_to_string(root.join("cpu.max")) {
        if let Some(n) = parse_cpu_max_text(&text) {
            return Some(n);
        }
        // "max" quota → unlimited; fall through to v1 only if v2 absent,
        // but the file existing means v2 is authoritative.
        if text.trim().starts_with("max") {
            return None;
        }
    }
    // v1: cfs_quota_us / cfs_period_us, quota -1 → unlimited.
    let v1 = root.join("cpu");
    let quota: i64 = fs::read_to_string(v1.join("cpu.cfs_quota_us"))
        .ok()?
        .trim()
        .parse()
        .ok()?;
    if quota <= 0 {
        return None;
    }
    let period: i64 = fs::read_to_string(v1.join("cpu.cfs_period_us"))
        .ok()?
        .trim()
        .parse()
        .ok()?;
    if period <= 0 {
        return None;
    }
    Some(div_ceil_u64(quota as u64, period as u64).max(1) as usize)
}

fn parse_cpu_max_text(text: &str) -> Option<usize> {
    let mut it = text.split_whitespace();
    let quota = it.next()?;
    if quota == "max" {
        return None;
    }
    let quota: u64 = quota.parse().ok()?;
    let period: u64 = it.next().unwrap_or("100000").parse().ok()?;
    if period == 0 {
        return None;
    }
    Some(div_ceil_u64(quota, period).max(1) as usize)
}

fn div_ceil_u64(a: u64, b: u64) -> u64 {
    a.div_ceil(b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn limit_text_parses_max_and_numbers() {
        assert_eq!(parse_limit_text("max\n"), None);
        assert_eq!(parse_limit_text(""), None);
        assert_eq!(parse_limit_text("4294967296\n"), Some(4 * GIB));
    }

    #[test]
    fn meminfo_field_parses_kib() {
        let text = "MemTotal:       16000000 kB\nMemAvailable:    8000000 kB\n";
        assert_eq!(
            parse_meminfo_field(text, "MemAvailable:"),
            Some(8_000_000 * KIB)
        );
        assert_eq!(
            parse_meminfo_field(text, "MemTotal:"),
            Some(16_000_000 * KIB)
        );
    }

    #[test]
    fn cpu_list_counts_ranges() {
        assert_eq!(parse_cpu_list_count("0-3,8"), 5);
        assert_eq!(parse_cpu_list_count("0"), 1);
        assert_eq!(parse_cpu_list_count(""), 0);
    }

    #[test]
    fn cpu_max_quota_rounds_up() {
        assert_eq!(parse_cpu_max_text("150000 100000\n"), Some(2));
        assert_eq!(parse_cpu_max_text("100000 100000\n"), Some(1));
        assert_eq!(parse_cpu_max_text("max 100000\n"), None);
    }

    #[test]
    fn cgroup_files_resolve_available_from_tree() {
        let dir = std::env::temp_dir().join(format!("tt-budget-test-{}", std::process::id()));
        let _ = fs::create_dir_all(&dir);
        fs::write(dir.join("memory.max"), "8589934592\n").unwrap();
        fs::write(dir.join("memory.current"), "2147483648\n").unwrap();
        let got = cgroup_memory_available(&dir).unwrap();
        assert_eq!(got, (6 * GIB, "cgroup_v2"));
        fs::write(dir.join("cpu.max"), "200000 100000\n").unwrap();
        assert_eq!(cgroup_cpu_quota(&dir), Some(2));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn reserve_is_clamped() {
        assert_eq!(memory_reserve_bytes(GIB), 256 * MIB);
        assert_eq!(memory_reserve_bytes(16 * GIB), 16 * GIB / 20);
        assert_eq!(memory_reserve_bytes(200 * GIB), 2 * GIB);
    }

    #[test]
    fn live_budget_is_sane() {
        let b = effective_host_budget();
        assert!(b.memory_bytes >= 128 * MIB);
        assert!(b.cpu_count >= 1);
        assert!(!b.memory_source.is_empty());
        assert!(!b.cpu_source.is_empty());
    }

    #[test]
    fn live_uncapped_os_budget_exceeds_floor() {
        let b = effective_host_budget();
        if !matches!(b.memory_source, "os_available" | "total_fallback") {
            return;
        }
        assert!(
            b.memory_bytes > 128 * MIB,
            "OS probe must exceed the 128 MiB floor; source={} bytes={}",
            b.memory_source,
            b.memory_bytes
        );
    }
}
