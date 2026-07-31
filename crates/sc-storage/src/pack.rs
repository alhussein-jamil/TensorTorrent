//! Versioned pack manifest + positional reader with bounds checks.

use crate::error::{StorageError, StorageResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
#[cfg(not(unix))]
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

pub const PACK_FORMAT_VERSION: u32 = 1;
const MAX_TENSOR_BYTES: u64 = 64 * 1024 * 1024 * 1024; // 64 GiB hard ceiling

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
        let mut seen = HashMap::new();
        for t in &self.tensors {
            if t.name.is_empty() {
                return Err(StorageError::Invalid("empty tensor name".into()));
            }
            if t.length > MAX_TENSOR_BYTES {
                return Err(StorageError::ExcessiveAllocation(t.length));
            }
            // Overflow check.
            let end = t
                .offset
                .checked_add(t.length)
                .ok_or_else(|| StorageError::Invalid(format!("offset overflow for {}", t.name)))?;
            if let Some((other, other_end)) = seen.insert(t.name.clone(), (t.offset, end)) {
                let _ = other;
                let _ = other_end;
                return Err(StorageError::Invalid(format!(
                    "duplicate tensor {}",
                    t.name
                )));
            }
            if t.dtype.is_empty() {
                return Err(StorageError::Invalid(format!(
                    "invalid dtype for {}",
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
        // Overlap detection (O(n^2) OK for modest manifests).
        for (i, a) in self.tensors.iter().enumerate() {
            let a_end = a.offset + a.length;
            for b in &self.tensors[i + 1..] {
                let b_end = b.offset + b.length;
                if a.offset < b_end && b.offset < a_end {
                    return Err(StorageError::Invalid(format!(
                        "overlapping ranges {} and {}",
                        a.name, b.name
                    )));
                }
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
        let mut buf = vec![0u8; length as usize];
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
}
