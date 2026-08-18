//! Activation spill files: Rust owns paths, writes, reads, cleanup.
//!
//! Payload is contiguous host bytes plus dtype/shape metadata. Python only
//! converts `torch.Tensor` ↔ bytes; it does not own spill bookkeeping.

use crate::error::{StorageError, StorageResult};
use serde::{Deserialize, Serialize};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

const MAGIC: &[u8; 8] = b"SCSPILL1";
const MAX_DTYPE_BYTES: usize = 256;
const MAX_TENSOR_RANK: usize = 64;
const MAX_SPILL_BYTES: u64 = 64 * 1024 * 1024 * 1024;
static NEXT_SPILL_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct SpillMeta {
    pub dtype: String,
    pub shape: Vec<i64>,
    pub nbytes: u64,
}

/// Write contiguous activation bytes to a new temp file under ``dir``.
pub fn write_activation_spill(
    dir: &Path,
    meta: &SpillMeta,
    bytes: &[u8],
) -> StorageResult<PathBuf> {
    if bytes.len() as u64 != meta.nbytes {
        return Err(StorageError::Io(format!(
            "spill nbytes mismatch: meta={} bytes={}",
            meta.nbytes,
            bytes.len()
        )));
    }
    if meta.dtype.is_empty() || meta.dtype.len() > MAX_DTYPE_BYTES {
        return Err(StorageError::Invalid("invalid spill dtype".into()));
    }
    if meta.shape.len() > MAX_TENSOR_RANK || meta.shape.iter().any(|dim| *dim < 0) {
        return Err(StorageError::Invalid("invalid spill shape".into()));
    }
    if meta.nbytes > MAX_SPILL_BYTES {
        return Err(StorageError::ExcessiveAllocation(meta.nbytes));
    }
    fs::create_dir_all(dir).map_err(|e| StorageError::Io(e.to_string()))?;
    // Refuse to fill the disk: writing must leave a safety margin so the
    // host filesystem (and our own manifest/log writes) keep working.
    if let Some(free) = free_space_bytes(dir) {
        let needed = meta.nbytes.saturating_add(SPILL_FREE_SPACE_MARGIN);
        if needed > free {
            return Err(StorageError::DiskSpace {
                path: dir.display().to_string(),
                needed: meta.nbytes,
                free,
            });
        }
    }
    let mut path = dir.to_path_buf();
    let name = format!(
        "tt_act_{}_{}_{}.spill",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0),
        NEXT_SPILL_ID.fetch_add(1, Ordering::Relaxed)
    );
    path.push(name);
    let tmp = path.with_extension("spill.tmp");
    let write_result = (|| -> StorageResult<()> {
        let mut f = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp)
            .map_err(|e| StorageError::Io(e.to_string()))?;
        f.write_all(MAGIC)
            .map_err(|e| StorageError::Io(e.to_string()))?;
        let dtype = meta.dtype.as_bytes();
        f.write_all(&(dtype.len() as u32).to_le_bytes())
            .map_err(|e| StorageError::Io(e.to_string()))?;
        f.write_all(dtype)
            .map_err(|e| StorageError::Io(e.to_string()))?;
        f.write_all(&(meta.shape.len() as u32).to_le_bytes())
            .map_err(|e| StorageError::Io(e.to_string()))?;
        for dim in &meta.shape {
            f.write_all(&dim.to_le_bytes())
                .map_err(|e| StorageError::Io(e.to_string()))?;
        }
        f.write_all(&meta.nbytes.to_le_bytes())
            .map_err(|e| StorageError::Io(e.to_string()))?;
        f.write_all(bytes)
            .map_err(|e| StorageError::Io(e.to_string()))?;
        f.sync_all().map_err(|e| StorageError::Io(e.to_string()))?;
        Ok(())
    })();
    if let Err(error) = write_result {
        let _ = fs::remove_file(&tmp);
        return Err(error);
    }
    if let Err(error) = fs::rename(&tmp, &path) {
        let _ = fs::remove_file(&tmp);
        return Err(StorageError::Io(error.to_string()));
    }
    Ok(path)
}

/// Read a spill file written by [`write_activation_spill`].
pub fn read_activation_spill(path: &Path) -> StorageResult<(SpillMeta, Vec<u8>)> {
    let mut f = File::open(path).map_err(|e| StorageError::Io(e.to_string()))?;
    let file_size = f
        .metadata()
        .map_err(|e| StorageError::Io(e.to_string()))?
        .len();
    let mut magic = [0u8; 8];
    f.read_exact(&mut magic)
        .map_err(|e| StorageError::Io(e.to_string()))?;
    if &magic != MAGIC {
        return Err(StorageError::Io(format!(
            "bad spill magic in {}",
            path.display()
        )));
    }
    let mut len_buf = [0u8; 4];
    f.read_exact(&mut len_buf)
        .map_err(|e| StorageError::Io(e.to_string()))?;
    let dtype_len = u32::from_le_bytes(len_buf) as usize;
    if dtype_len == 0 || dtype_len > MAX_DTYPE_BYTES {
        return Err(StorageError::Invalid("invalid spill dtype length".into()));
    }
    let mut dtype_bytes = Vec::new();
    dtype_bytes
        .try_reserve_exact(dtype_len)
        .map_err(|_| StorageError::ExcessiveAllocation(dtype_len as u64))?;
    dtype_bytes.resize(dtype_len, 0);
    f.read_exact(&mut dtype_bytes)
        .map_err(|e| StorageError::Io(e.to_string()))?;
    let dtype = String::from_utf8(dtype_bytes)
        .map_err(|e| StorageError::Io(format!("spill dtype utf8: {e}")))?;
    f.read_exact(&mut len_buf)
        .map_err(|e| StorageError::Io(e.to_string()))?;
    let ndim = u32::from_le_bytes(len_buf) as usize;
    if ndim > MAX_TENSOR_RANK {
        return Err(StorageError::Invalid("invalid spill tensor rank".into()));
    }
    let mut shape = Vec::with_capacity(ndim);
    for _ in 0..ndim {
        let mut dim_buf = [0u8; 8];
        f.read_exact(&mut dim_buf)
            .map_err(|e| StorageError::Io(e.to_string()))?;
        let dim = i64::from_le_bytes(dim_buf);
        if dim < 0 {
            return Err(StorageError::Invalid(
                "negative spill tensor dimension".into(),
            ));
        }
        shape.push(dim);
    }
    let mut nb = [0u8; 8];
    f.read_exact(&mut nb)
        .map_err(|e| StorageError::Io(e.to_string()))?;
    let nbytes = u64::from_le_bytes(nb);
    if nbytes > MAX_SPILL_BYTES {
        return Err(StorageError::ExcessiveAllocation(nbytes));
    }
    let header_size = 8u64
        .checked_add(4)
        .and_then(|size| size.checked_add(dtype_len as u64))
        .and_then(|size| size.checked_add(4))
        .and_then(|size| size.checked_add((ndim as u64).saturating_mul(8)))
        .and_then(|size| size.checked_add(8))
        .ok_or_else(|| StorageError::Invalid("spill header size overflow".into()))?;
    let expected_size = header_size
        .checked_add(nbytes)
        .ok_or_else(|| StorageError::Invalid("spill file size overflow".into()))?;
    if expected_size != file_size {
        return Err(StorageError::Invalid(format!(
            "spill file size mismatch: encoded={expected_size} actual={file_size}"
        )));
    }
    let allocation =
        usize::try_from(nbytes).map_err(|_| StorageError::ExcessiveAllocation(nbytes))?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(allocation)
        .map_err(|_| StorageError::ExcessiveAllocation(nbytes))?;
    bytes.resize(allocation, 0);
    f.read_exact(&mut bytes)
        .map_err(|e| StorageError::Io(e.to_string()))?;
    Ok((
        SpillMeta {
            dtype,
            shape,
            nbytes,
        },
        bytes,
    ))
}

pub fn remove_activation_spill(path: &Path) -> StorageResult<()> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(StorageError::Io(e.to_string())),
    }
}

/// Prefix for per-execution spill session directories:
/// `tt-spill-<pid>-<execution_id>`. The pid embedding lets a later process
/// prove ownership is dead before sweeping.
pub const SPILL_SESSION_PREFIX: &str = "tt-spill-";
/// Free-space margin the spill writer always leaves on the filesystem.
pub const SPILL_FREE_SPACE_MARGIN: u64 = 64 * 1024 * 1024;

/// Free bytes available to unprivileged writes at `path` (statvfs), if known.
#[must_use]
// statvfs field widths differ across libc targets; the casts are portability-required.
#[allow(clippy::unnecessary_cast)]
pub fn free_space_bytes(path: &Path) -> Option<u64> {
    use std::os::unix::ffi::OsStrExt;
    let c = std::ffi::CString::new(path.as_os_str().as_bytes()).ok()?;
    // SAFETY: statvfs writes into the zeroed struct on success; the path
    // pointer is a valid NUL-terminated C string for the duration of the call.
    unsafe {
        let mut st: libc::statvfs = std::mem::zeroed();
        if libc::statvfs(c.as_ptr(), &mut st) != 0 {
            return None;
        }
        Some((st.f_bavail as u64).saturating_mul(st.f_frsize as u64))
    }
}

/// True when `path` sits on a RAM-backed filesystem (tmpfs/ramfs).
///
/// Spilling activations there writes into the very memory spill exists to
/// relieve; on desktop Linux `/tmp` is usually tmpfs sized at 50% of RAM.
#[must_use]
// statfs::f_type is c_long/i64 depending on target; the cast is portability-required.
#[allow(clippy::unnecessary_cast)]
pub fn is_ram_backed_fs(path: &Path) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        let Ok(c) = std::ffi::CString::new(path.as_os_str().as_bytes()) else {
            return false;
        };
        // SAFETY: statfs writes into the zeroed struct on success; path pointer
        // is valid for the call.
        unsafe {
            let mut st: libc::statfs = std::mem::zeroed();
            if libc::statfs(c.as_ptr(), &mut st) != 0 {
                return false;
            }
            ram_backed_statfs(&st)
        }
    }
    #[cfg(not(unix))]
    {
        let _ = path;
        false
    }
}

#[cfg(target_os = "linux")]
fn ram_backed_statfs(st: &libc::statfs) -> bool {
    const TMPFS_MAGIC: i64 = 0x0102_1994;
    const RAMFS_MAGIC: i64 = 0x8584_58f6;
    let ftype = st.f_type as i64;
    ftype == TMPFS_MAGIC || ftype == RAMFS_MAGIC
}

#[cfg(target_os = "macos")]
fn ram_backed_statfs(st: &libc::statfs) -> bool {
    let name = unsafe { std::ffi::CStr::from_ptr(st.f_fstypename.as_ptr()) };
    matches!(name.to_str().unwrap_or(""), "tmpfs" | "ramfs")
}

#[cfg(all(unix, not(any(target_os = "linux", target_os = "macos"))))]
fn ram_backed_statfs(_st: &libc::statfs) -> bool {
    false
}

/// Validate a spill base directory: creatable, and not RAM-backed unless
/// explicitly allowed (tests set `allow_ram_backed`).
pub fn ensure_spill_dir_usable(dir: &Path, allow_ram_backed: bool) -> StorageResult<()> {
    fs::create_dir_all(dir).map_err(|e| StorageError::Io(e.to_string()))?;
    if !allow_ram_backed && is_ram_backed_fs(dir) {
        return Err(StorageError::SpillDirUnsuitable(format!(
            "{} is on a RAM-backed filesystem (tmpfs/ramfs); spilling there consumes \
             the memory spill is meant to relieve. Configure spill_dir (or TT_SPILL_DIR) \
             on a real block device, or set TT_ALLOW_TMPFS_SPILL=1 to override",
            dir.display()
        )));
    }
    Ok(())
}

/// Create the per-execution session directory under `base`.
pub fn create_spill_session_dir(base: &Path, execution_id: u64) -> StorageResult<PathBuf> {
    let dir = base.join(format!(
        "{SPILL_SESSION_PREFIX}{}-{execution_id}",
        std::process::id()
    ));
    fs::create_dir_all(&dir).map_err(|e| StorageError::Io(e.to_string()))?;
    Ok(dir)
}

/// Best-effort removal of a session directory and its contents.
pub fn remove_spill_session_dir(dir: &Path) {
    let _ = fs::remove_dir_all(dir);
}

/// Remove session directories whose owning process is gone. Returns the
/// number of directories removed. Only exact `tt-spill-<pid>-<exec>` names
/// are touched; the current process's own sessions are never removed.
pub fn sweep_orphan_spill_sessions(base: &Path) -> usize {
    let Ok(entries) = fs::read_dir(base) else {
        return 0;
    };
    let own_pid = std::process::id();
    let mut removed = 0usize;
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(pid) = parse_session_owner_pid(&name.to_string_lossy()) else {
            continue;
        };
        if pid == own_pid || process_alive(pid) {
            continue;
        }
        if fs::remove_dir_all(entry.path()).is_ok() {
            removed += 1;
        }
    }
    removed
}

/// Parse the owning pid out of a `tt-spill-<pid>-<exec>` directory name.
fn parse_session_owner_pid(name: &str) -> Option<u32> {
    let rest = name.strip_prefix(SPILL_SESSION_PREFIX)?;
    let (pid_str, exec_str) = rest.split_once('-')?;
    if exec_str.is_empty() || !exec_str.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    pid_str.parse().ok()
}

fn process_alive(pid: u32) -> bool {
    // SAFETY: kill with signal 0 performs only a liveness/permission check.
    let rc = unsafe { libc::kill(pid as libc::pid_t, 0) };
    if rc == 0 {
        return true;
    }
    // EPERM means the process exists but is not ours: still alive.
    std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn roundtrip_spill_file() {
        let dir = env::temp_dir().join(format!("tt_spill_test_{}", std::process::id()));
        let meta = SpillMeta {
            dtype: "float32".into(),
            shape: vec![2, 3],
            nbytes: 24,
        };
        let payload = vec![1u8; 24];
        let path = write_activation_spill(&dir, &meta, &payload).unwrap();
        let (got_meta, got) = read_activation_spill(&path).unwrap();
        assert_eq!(got_meta, meta);
        assert_eq!(got, payload);
        remove_activation_spill(&path).unwrap();
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn rejects_excessive_dtype_length_before_allocation() {
        let dir = env::temp_dir().join(format!("tt_spill_bad_test_{}", std::process::id()));
        let _ = fs::create_dir_all(&dir);
        let path = dir.join("bad.spill");
        let mut file = File::create(&path).unwrap();
        file.write_all(MAGIC).unwrap();
        file.write_all(&u32::MAX.to_le_bytes()).unwrap();
        drop(file);
        assert!(read_activation_spill(&path).is_err());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn concurrent_spill_names_are_unique() {
        let dir = env::temp_dir().join(format!("tt_spill_unique_test_{}", std::process::id()));
        let meta = SpillMeta {
            dtype: "u8".into(),
            shape: vec![1],
            nbytes: 1,
        };
        let first = write_activation_spill(&dir, &meta, &[1]).unwrap();
        let second = write_activation_spill(&dir, &meta, &[2]).unwrap();
        assert_ne!(first, second);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn session_owner_pid_parses_only_exact_pattern() {
        assert_eq!(parse_session_owner_pid("tt-spill-123-7"), Some(123));
        assert_eq!(parse_session_owner_pid("tt-spill-123-"), None);
        assert_eq!(parse_session_owner_pid("tt-spill-abc-7"), None);
        assert_eq!(parse_session_owner_pid("something-else"), None);
        assert_eq!(parse_session_owner_pid("tt-spill-123-x9"), None);
    }

    #[test]
    fn sweep_removes_dead_owner_sessions_and_keeps_live_ones() {
        let base = env::temp_dir().join(format!("tt_sweep_test_{}", std::process::id()));
        let _ = fs::remove_dir_all(&base);
        fs::create_dir_all(&base).unwrap();
        // Our own live session must survive.
        let own = create_spill_session_dir(&base, 42).unwrap();
        // A dead-pid session must be removed (pid near u32 max cannot exist).
        let dead = base.join(format!("{SPILL_SESSION_PREFIX}4294000000-1"));
        fs::create_dir_all(&dead).unwrap();
        fs::write(dead.join("leftover.spill"), b"x").unwrap();
        // A non-matching directory must never be touched.
        let unrelated = base.join("user-data");
        fs::create_dir_all(&unrelated).unwrap();
        let removed = sweep_orphan_spill_sessions(&base);
        assert_eq!(removed, 1);
        assert!(own.is_dir());
        assert!(!dead.exists());
        assert!(unrelated.is_dir());
        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn ram_backed_fs_is_detected_and_refused() {
        let shm = Path::new("/dev/shm");
        if !shm.is_dir() {
            return; // environment without /dev/shm; covered in CI
        }
        assert!(is_ram_backed_fs(shm));
        let dir = shm.join(format!("tt_spill_tmpfs_{}", std::process::id()));
        let err = ensure_spill_dir_usable(&dir, false).unwrap_err();
        assert!(matches!(err, StorageError::SpillDirUnsuitable(_)));
        // Escape hatch for tests and RAM-disk-aware users.
        ensure_spill_dir_usable(&dir, true).unwrap();
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn free_space_probe_reports_something_sane() {
        let free = free_space_bytes(&env::temp_dir());
        assert!(free.is_some());
    }
}
