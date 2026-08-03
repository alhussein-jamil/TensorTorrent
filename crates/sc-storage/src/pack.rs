//! Versioned pack manifest + positional reader with bounds checks.

use crate::error::{StorageError, StorageResult};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs::File;
#[cfg(not(unix))]
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

pub const PACK_FORMAT_VERSION: u32 = 1;
const MAX_TENSOR_BYTES: u64 = 64 * 1024 * 1024 * 1024; // 64 GiB hard ceiling
const MAX_MANIFEST_JSON_BYTES: usize = 64 * 1024 * 1024;
const MAX_MANIFEST_TENSORS: usize = 100_000;
const MAX_TENSOR_NAME_BYTES: usize = 4096;
const MAX_DTYPE_BYTES: usize = 128;
const MAX_TENSOR_RANK: usize = 64;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TensorEntry {
    pub name: String,
    pub offset: u64,
    pub length: u64,
    pub dtype: String,
    pub shape: Vec<i64>,
    #[serde(default)]
    pub checksum_crc32: Option<u32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PackManifest {
    pub version: u32,
    pub tensors: Vec<TensorEntry>,
    #[serde(default)]
    pub notes: Vec<String>,
}

impl PackManifest {
    pub fn from_json(s: &str) -> StorageResult<Self> {
        if s.len() > MAX_MANIFEST_JSON_BYTES {
            return Err(StorageError::Invalid(format!(
                "manifest JSON exceeds {MAX_MANIFEST_JSON_BYTES} bytes"
            )));
        }
        let m: Self = serde_json::from_str(s)
            .map_err(|e| StorageError::Invalid(format!("manifest JSON: {e}")))?;
        m.validate()?;
        Ok(m)
    }

    pub fn validate(&self) -> StorageResult<()> {
        if self.version != PACK_FORMAT_VERSION {
            return Err(StorageError::Invalid(format!(
                "unsupported pack version {} (expected {PACK_FORMAT_VERSION})",
                self.version
            )));
        }
        if self.tensors.len() > MAX_MANIFEST_TENSORS {
            return Err(StorageError::Invalid(format!(
                "manifest has too many tensors: {} (max {MAX_MANIFEST_TENSORS})",
                self.tensors.len()
            )));
        }
        let mut seen = HashSet::with_capacity(self.tensors.len());
        let mut ranges = Vec::with_capacity(self.tensors.len());
        for t in &self.tensors {
            if t.name.is_empty() {
                return Err(StorageError::Invalid("empty tensor name".into()));
            }
            if t.name.len() > MAX_TENSOR_NAME_BYTES {
                return Err(StorageError::Invalid(format!(
                    "tensor name exceeds {MAX_TENSOR_NAME_BYTES} bytes"
                )));
            }
            if t.length > MAX_TENSOR_BYTES {
                return Err(StorageError::ExcessiveAllocation(t.length));
            }
            // Overflow check.
            let end = t
                .offset
                .checked_add(t.length)
                .ok_or_else(|| StorageError::Invalid(format!("offset overflow for {}", t.name)))?;
            if !seen.insert(t.name.as_str()) {
                return Err(StorageError::Invalid(format!(
                    "duplicate tensor {}",
                    t.name
                )));
            }
            ranges.push((t.offset, end, t.name.as_str()));
            if t.dtype.is_empty() || t.dtype.len() > MAX_DTYPE_BYTES {
                return Err(StorageError::Invalid(format!(
                    "invalid dtype for {}",
                    t.name
                )));
            }
            if t.shape.len() > MAX_TENSOR_RANK {
                return Err(StorageError::Invalid(format!(
                    "tensor rank exceeds {MAX_TENSOR_RANK} for {}",
                    t.name
                )));
            }
            for d in &t.shape {
                if *d < 0 {
                    return Err(StorageError::Invalid(format!(
                        "invalid shape for {}",
                        t.name
                    )));
                }
            }
        }
        ranges.sort_unstable_by_key(|(offset, end, _)| (*offset, *end));
        let mut active: Option<(u64, &str)> = None;
        for (offset, end, name) in ranges {
            if let Some((active_end, active_name)) = active {
                if offset < active_end && offset < end {
                    return Err(StorageError::Invalid(format!(
                        "overlapping ranges {active_name} and {name}"
                    )));
                }
            }
            if offset < end && active.map_or(true, |(active_end, _)| end > active_end) {
                active = Some((end, name));
            }
        }
        Ok(())
    }
}

pub struct PackReader {
    path: PathBuf,
    file: File,
    file_size: u64,
    manifest: PackManifest,
    index: HashMap<String, usize>,
}

impl PackReader {
    pub fn open(path: impl AsRef<Path>, manifest: PackManifest) -> StorageResult<Self> {
        manifest.validate()?;
        let path = path.as_ref().to_path_buf();
        let path_metadata =
            std::fs::symlink_metadata(&path).map_err(|e| StorageError::Io(e.to_string()))?;
        if path_metadata.file_type().is_symlink() {
            return Err(StorageError::Invalid(format!(
                "pack path cannot be a symlink: {}",
                path.display()
            )));
        }
        if !path_metadata.is_file() {
            return Err(StorageError::Invalid(format!(
                "pack path is not a regular file: {}",
                path.display()
            )));
        }
        let file = File::open(&path).map_err(|e| StorageError::Io(e.to_string()))?;
        let file_size = file
            .metadata()
            .map_err(|e| StorageError::Io(e.to_string()))?
            .len();
        for t in &manifest.tensors {
            let end = t.offset.checked_add(t.length).ok_or(StorageError::Bounds {
                offset: t.offset,
                length: t.length,
                file_size,
            })?;
            if end > file_size {
                return Err(StorageError::Bounds {
                    offset: t.offset,
                    length: t.length,
                    file_size,
                });
            }
        }
        let index = manifest
            .tensors
            .iter()
            .enumerate()
            .map(|(i, t)| (t.name.clone(), i))
            .collect();
        Ok(Self {
            path,
            file,
            file_size,
            manifest,
            index,
        })
    }

    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    #[must_use]
    pub fn manifest(&self) -> &PackManifest {
        &self.manifest
    }

    pub fn pread(&mut self, name: &str) -> StorageResult<Vec<u8>> {
        let idx = *self
            .index
            .get(name)
            .ok_or_else(|| StorageError::Invalid(format!("unknown tensor {name}")))?;
        let entry = &self.manifest.tensors[idx];
        let offset = entry.offset;
        let length = entry.length;
        if length > MAX_TENSOR_BYTES {
            return Err(StorageError::ExcessiveAllocation(length));
        }
        let end = offset.checked_add(length).ok_or(StorageError::Bounds {
            offset,
            length,
            file_size: self.file_size,
        })?;
        if end > self.file_size {
            return Err(StorageError::Bounds {
                offset,
                length,
                file_size: self.file_size,
            });
        }
        let allocation =
            usize::try_from(length).map_err(|_| StorageError::ExcessiveAllocation(length))?;
        let mut buf = Vec::new();
        buf.try_reserve_exact(allocation)
            .map_err(|_| StorageError::ExcessiveAllocation(length))?;
        buf.resize(allocation, 0);
        // Prefer positional I/O so concurrent workers never contend on seek.
        #[cfg(unix)]
        {
            use std::os::unix::fs::FileExt;
            self.file
                .read_exact_at(&mut buf, offset)
                .map_err(|e| StorageError::Io(e.to_string()))?;
        }
        #[cfg(not(unix))]
        {
            self.file
                .seek(SeekFrom::Start(offset))
                .map_err(|e| StorageError::Io(e.to_string()))?;
            self.file
                .read_exact(&mut buf)
                .map_err(|e| StorageError::Io(e.to_string()))?;
        }
        if let Some(expected) = entry.checksum_crc32 {
            let got = crc32_ieee(&buf);
            if got != expected {
                return Err(StorageError::ChecksumMismatch {
                    tensor: name.to_owned(),
                    expected: format!("{expected:#x}"),
                    got: format!("{got:#x}"),
                });
            }
        }
        Ok(buf)
    }
}

/// Incremental IEEE CRC32.
#[must_use]
pub fn crc32_ieee(data: &[u8]) -> u32 {
    let mut crc = 0xffff_ffffu32;
    for &b in data {
        crc ^= u32::from(b);
        for _ in 0..8 {
            let mask = (!(crc & 1)).wrapping_add(1); // 0 or 0xFFFFFFFF
            crc = (crc >> 1) ^ (0xEDB8_8320u32 & mask);
        }
    }
    !crc
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn rejects_overlap() {
        let m = PackManifest {
            version: 1,
            tensors: vec![
                TensorEntry {
                    name: "a".into(),
                    offset: 0,
                    length: 10,
                    dtype: "f32".into(),
                    shape: vec![2, 1],
                    checksum_crc32: None,
                },
                TensorEntry {
                    name: "b".into(),
                    offset: 5,
                    length: 10,
                    dtype: "f32".into(),
                    shape: vec![2, 1],
                    checksum_crc32: None,
                },
            ],
            notes: vec![],
        };
        assert!(m.validate().is_err());
    }

    #[test]
    fn rejects_overlap_separated_by_zero_length_entry() {
        let tensor = |name: &str, offset: u64, length: u64| TensorEntry {
            name: name.into(),
            offset,
            length,
            dtype: "u8".into(),
            shape: vec![length as i64],
            checksum_crc32: None,
        };
        let manifest = PackManifest {
            version: PACK_FORMAT_VERSION,
            tensors: vec![
                tensor("outer", 0, 100),
                tensor("empty", 10, 0),
                tensor("inner", 20, 1),
            ],
            notes: vec![],
        };
        assert!(manifest.validate().is_err());
    }

    #[test]
    fn rejects_excessive_tensor_rank() {
        let manifest = PackManifest {
            version: PACK_FORMAT_VERSION,
            tensors: vec![TensorEntry {
                name: "w".into(),
                offset: 0,
                length: 1,
                dtype: "u8".into(),
                shape: vec![1; MAX_TENSOR_RANK + 1],
                checksum_crc32: None,
            }],
            notes: vec![],
        };
        assert!(manifest.validate().is_err());
    }

    #[test]
    fn pread_roundtrip() {
        let dir = std::env::temp_dir().join("sc-pack-test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("data.bin");
        let payload = b"hello-streamcompiler";
        {
            let mut f = File::create(&path).unwrap();
            f.write_all(payload).unwrap();
        }
        let crc = crc32_ieee(payload);
        let m = PackManifest {
            version: 1,
            tensors: vec![TensorEntry {
                name: "w".into(),
                offset: 0,
                length: payload.len() as u64,
                dtype: "u8".into(),
                shape: vec![payload.len() as i64],
                checksum_crc32: Some(crc),
            }],
            notes: vec![],
        };
        let mut reader = PackReader::open(&path, m).unwrap();
        let got = reader.pread("w").unwrap();
        assert_eq!(got, payload);
    }

    #[cfg(unix)]
    #[test]
    fn rejects_symlink_pack_path() {
        use std::os::unix::fs::symlink;

        let dir = std::env::temp_dir().join(format!(
            "sc-pack-symlink-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("data.bin");
        std::fs::write(&path, b"1234").unwrap();
        let manifest = PackManifest {
            version: PACK_FORMAT_VERSION,
            tensors: vec![TensorEntry {
                name: "w".into(),
                offset: 0,
                length: 4,
                dtype: "u8".into(),
                shape: vec![4],
                checksum_crc32: None,
            }],
            notes: vec![],
        };
        let link = path.with_file_name("data-link.bin");
        symlink(&path, &link).unwrap();
        let error = PackReader::open(&link, manifest).err().unwrap();
        assert!(error.to_string().contains("cannot be a symlink"));
    }
}
