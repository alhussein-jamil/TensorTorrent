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
    let mut path = dir.to_path_buf();
    let name = format!(
        "sc_act_{}_{}_{}.spill",
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn roundtrip_spill_file() {
        let dir = env::temp_dir().join(format!("sc_spill_test_{}", std::process::id()));
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
        let dir = env::temp_dir().join(format!("sc_spill_bad_test_{}", std::process::id()));
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
        let dir = env::temp_dir().join(format!("sc_spill_unique_test_{}", std::process::id()));
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
}
