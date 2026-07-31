//! Activation spill files: Rust owns paths, writes, reads, cleanup.
//!
//! Payload is contiguous host bytes plus dtype/shape metadata. Python only
//! converts `torch.Tensor` ↔ bytes; it does not own spill bookkeeping.

use crate::error::{StorageError, StorageResult};
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

const MAGIC: &[u8; 8] = b"SCSPILL1";

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
    fs::create_dir_all(dir).map_err(|e| StorageError::Io(e.to_string()))?;
    let mut path = dir.to_path_buf();
    let name = format!(
        "sc_act_{}.spill",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    );
    path.push(name);
    let tmp = path.with_extension("spill.tmp");
    {
        let mut f = File::create(&tmp).map_err(|e| StorageError::Io(e.to_string()))?;
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
    }
    fs::rename(&tmp, &path).map_err(|e| StorageError::Io(e.to_string()))?;
    Ok(path)
}

/// Read a spill file written by [`write_activation_spill`].
pub fn read_activation_spill(path: &Path) -> StorageResult<(SpillMeta, Vec<u8>)> {
    let mut f = File::open(path).map_err(|e| StorageError::Io(e.to_string()))?;
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
    let mut dtype_bytes = vec![0u8; dtype_len];
    f.read_exact(&mut dtype_bytes)
        .map_err(|e| StorageError::Io(e.to_string()))?;
    let dtype = String::from_utf8(dtype_bytes)
        .map_err(|e| StorageError::Io(format!("spill dtype utf8: {e}")))?;
    f.read_exact(&mut len_buf)
        .map_err(|e| StorageError::Io(e.to_string()))?;
    let ndim = u32::from_le_bytes(len_buf) as usize;
    let mut shape = Vec::with_capacity(ndim);
    for _ in 0..ndim {
        let mut dim_buf = [0u8; 8];
        f.read_exact(&mut dim_buf)
            .map_err(|e| StorageError::Io(e.to_string()))?;
        shape.push(i64::from_le_bytes(dim_buf));
    }
    let mut nb = [0u8; 8];
    f.read_exact(&mut nb)
        .map_err(|e| StorageError::Io(e.to_string()))?;
    let nbytes = u64::from_le_bytes(nb);
    let mut bytes = vec![0u8; nbytes as usize];
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
}
