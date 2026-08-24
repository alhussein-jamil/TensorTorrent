//! OS memory and CPU probes. Linux uses procfs; other Unix hosts use sysctl.
//!
//! Callers (budget + NUMA) must not read `/proc` or `/sys` directly.

use std::thread;

#[cfg(target_os = "linux")]
use std::fs;
#[cfg(target_os = "linux")]
use std::path::Path;
#[cfg(target_os = "linux")]
const KIB: u64 = 1024;

/// Live remaining host RAM, when the OS exposes it.
#[must_use]
pub fn memory_available_bytes() -> Option<u64> {
    os::memory_available_bytes()
}

/// Physical host RAM, when the OS exposes it.
#[must_use]
pub fn memory_total_bytes() -> Option<u64> {
    os::memory_total_bytes()
}

/// Logical CPUs this process can run on (`available_parallelism`).
#[must_use]
pub fn online_cpu_count() -> usize {
    thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(1)
        .max(1)
}

/// Physical packages / sockets. ``1`` when the OS does not expose the count.
#[must_use]
pub fn socket_count() -> u32 {
    os::socket_count().max(1)
}

#[cfg(target_os = "linux")]
mod os {
    use super::{fs, Path, KIB};

    pub(super) fn memory_available_bytes() -> Option<u64> {
        parse_meminfo_field(
            &fs::read_to_string(Path::new("/proc/meminfo")).ok()?,
            "MemAvailable:",
        )
    }

    pub(super) fn memory_total_bytes() -> Option<u64> {
        parse_meminfo_field(
            &fs::read_to_string(Path::new("/proc/meminfo")).ok()?,
            "MemTotal:",
        )
    }

    pub(super) fn parse_meminfo_field(text: &str, field: &str) -> Option<u64> {
        for line in text.lines() {
            if let Some(rest) = line.strip_prefix(field) {
                let kb: u64 = rest.split_whitespace().next()?.parse().ok()?;
                return Some(kb.saturating_mul(KIB));
            }
        }
        None
    }

    pub(super) fn socket_count() -> u32 {
        let Ok(text) = fs::read_to_string(Path::new("/proc/cpuinfo")) else {
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
}

#[cfg(target_os = "macos")]
mod os {
    use std::ffi::CStr;
    use std::ptr;

    fn sysctl_u64(name: &CStr) -> Option<u64> {
        let mut len = 0usize;
        let rc = unsafe {
            libc::sysctlbyname(name.as_ptr(), ptr::null_mut(), &mut len, ptr::null_mut(), 0)
        };
        if rc != 0 || len == 0 || len > 8 {
            return None;
        }
        let mut buf = [0u8; 8];
        let mut n = len;
        let rc = unsafe {
            libc::sysctlbyname(
                name.as_ptr(),
                buf.as_mut_ptr().cast::<libc::c_void>(),
                &mut n,
                ptr::null_mut(),
                0,
            )
        };
        if rc != 0 {
            return None;
        }
        match n {
            4 => Some(u32::from_ne_bytes(buf[..4].try_into().ok()?) as u64),
            8 => Some(u64::from_ne_bytes(buf[..8].try_into().ok()?)),
            _ => None,
        }
    }

    pub(super) fn memory_total_bytes() -> Option<u64> {
        sysctl_u64(c"hw.memsize")
    }

    pub(super) fn memory_available_bytes() -> Option<u64> {
        let page = sysctl_u64(c"hw.pagesize")?;
        if page == 0 {
            return None;
        }
        let free = sysctl_u64(c"vm.page_free_count").unwrap_or(0);
        let inactive = sysctl_u64(c"vm.page_inactive_count").unwrap_or(0);
        let speculative = sysctl_u64(c"vm.page_speculative_count").unwrap_or(0);
        let pages = free.saturating_add(inactive).saturating_add(speculative);
        if pages == 0 {
            return memory_total_bytes();
        }
        Some(pages.saturating_mul(page))
    }

    pub(super) fn socket_count() -> u32 {
        sysctl_u64(c"hw.packages").unwrap_or(1).max(1) as u32
    }
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
mod os {
    pub(super) fn memory_available_bytes() -> Option<u64> {
        None
    }

    pub(super) fn memory_total_bytes() -> Option<u64> {
        None
    }

    pub(super) fn socket_count() -> u32 {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_count_is_at_least_one() {
        assert!(online_cpu_count() >= 1);
        assert!(socket_count() >= 1);
    }

    #[test]
    fn live_os_exposes_memory_on_supported_unix() {
        let total = memory_total_bytes();
        let avail = memory_available_bytes();
        assert!(
            total.unwrap_or(0) >= 128 * 1024 * 1024 || avail.unwrap_or(0) >= 128 * 1024 * 1024,
            "OS memory probe empty: total={total:?} available={avail:?}"
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn meminfo_field_parses_kib() {
        let text = "MemTotal:       16000000 kB\nMemAvailable:    8000000 kB\n";
        assert_eq!(
            os::parse_meminfo_field(text, "MemAvailable:"),
            Some(8_000_000 * KIB)
        );
        assert_eq!(
            os::parse_meminfo_field(text, "MemTotal:"),
            Some(16_000_000 * KIB)
        );
    }
}
