//! Logical tensor residency: exact-resource copies, versions, leases, aliases.

use crate::allocation::AllocationTable;
use crate::error::{MemoryError, MemoryResult};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use streamcompiler_core::{AllocationId, EventId, ResourceId, TensorId};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TensorMetadata {
    pub shape: Vec<i64>,
    pub strides: Vec<i64>,
    pub storage_offset: i64,
    pub dtype: String,
    pub nbytes: u64,
    pub alias_group: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ResidentCopy {
    pub resource: ResourceId,
    pub allocation: AllocationId,
    pub version: u64,
    pub ready_event: Option<EventId>,
    pub active_leases: u32,
    pub valid: bool,
    pub authoritative: bool,
    pub storage_offset: i64,
    pub shape: Vec<i64>,
    pub strides: Vec<i64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LogicalTensorRecord {
    pub id: TensorId,
    pub version: u64,
    pub metadata: TensorMetadata,
    pub alias_group: Option<String>,
    pub copies: HashMap<String, ResidentCopy>,
}

/// Authoritative multi-copy residency store.
#[derive(Clone, Debug, Default)]
pub struct ResidencyStore {
    inner: Arc<Mutex<Inner>>,
    allocations: Arc<AllocationTable>,
}

#[derive(Debug, Default)]
struct Inner {
    tensors: HashMap<String, LogicalTensorRecord>,
    /// In-flight transfer dedupe: (tensor, dest_resource) -> transfer instruction id.
    inflight_transfers: HashMap<(String, String), String>,
}

impl ResidencyStore {
    #[must_use]
    pub fn new(allocations: Arc<AllocationTable>) -> Self {
        Self {
            inner: Arc::new(Mutex::new(Inner::default())),
            allocations,
        }
    }

    #[must_use]
    pub fn allocations(&self) -> Arc<AllocationTable> {
        Arc::clone(&self.allocations)
    }

    /// Materialize or mutate on `resource` — bumps logical version, invalidates siblings.
    pub fn put(
        &self,
        tensor: TensorId,
        resource: ResourceId,
        allocation: AllocationId,
        metadata: TensorMetadata,
        ready_event: Option<EventId>,
    ) -> MemoryResult<ResidentCopy> {
        let nbytes = metadata.nbytes;
        self.allocations
            .register(allocation.clone(), resource.clone(), nbytes.max(1), 64)?;
        let mut g = self.inner.lock();
        let tid = tensor.as_str().to_owned();
        let version = g.tensors.get(&tid).map(|t| t.version + 1).unwrap_or(1);
        let alias = metadata.alias_group.clone();
        let entry = g
            .tensors
            .entry(tid.clone())
            .or_insert_with(|| LogicalTensorRecord {
                id: tensor.clone(),
                version: 0,
                metadata: metadata.clone(),
                alias_group: alias.clone(),
                copies: HashMap::new(),
            });
        entry.version = version;
        entry.metadata = metadata.clone();
        entry.alias_group = alias;
        for (rid, copy) in entry.copies.iter_mut() {
            if rid != resource.as_str() {
                copy.valid = false;
                copy.authoritative = false;
            }
        }
        // Replace same-key: release prior allocation ref if different.
        if let Some(prev) = entry.copies.get(resource.as_str()) {
            if prev.allocation != allocation {
                let _ = self.allocations.release(&prev.allocation);
            } else {
                // Same allocation re-put: undo the extra register bump below... already registered.
                // We registered once above; if same alloc, register bumped ref — compensate by releasing once.
                let _ = self.allocations.release(&allocation);
            }
        }
        let copy = ResidentCopy {
            resource: resource.clone(),
            allocation,
            version,
            ready_event,
            active_leases: 0,
            valid: true,
            authoritative: true,
            storage_offset: metadata.storage_offset,
            shape: metadata.shape.clone(),
            strides: metadata.strides.clone(),
        };
        entry
            .copies
            .insert(resource.as_str().to_owned(), copy.clone());
        Ok(copy)
    }

    /// Immutable cross-resource copy — same logical version, no sibling invalidation.
    pub fn replicate(
        &self,
        tensor: &TensorId,
        resource: ResourceId,
        allocation: AllocationId,
        ready_event: Option<EventId>,
    ) -> MemoryResult<ResidentCopy> {
        let mut g = self.inner.lock();
        let entry = g
            .tensors
            .get_mut(tensor.as_str())
            .ok_or_else(|| MemoryError::NotResident {
                tensor: tensor.to_string(),
                resource: resource.to_string(),
            })?;
        let version = entry.version;
        let meta = entry.metadata.clone();
        drop(g);
        self.allocations
            .register(allocation.clone(), resource.clone(), meta.nbytes.max(1), 64)?;
        let mut g = self.inner.lock();
        let entry = g.tensors.get_mut(tensor.as_str()).unwrap();
        if let Some(prev) = entry.copies.get(resource.as_str()) {
            if prev.allocation == allocation {
                let _ = self.allocations.release(&allocation);
            } else {
                let _ = self.allocations.release(&prev.allocation);
            }
        }
        let copy = ResidentCopy {
            resource: resource.clone(),
            allocation,
            version,
            ready_event,
            active_leases: 0,
            valid: true,
            authoritative: false,
            storage_offset: meta.storage_offset,
            shape: meta.shape,
            strides: meta.strides,
        };
        entry
            .copies
            .insert(resource.as_str().to_owned(), copy.clone());
        Ok(copy)
    }

    pub fn get(&self, tensor: &TensorId, resource: &ResourceId) -> MemoryResult<ResidentCopy> {
        let g = self.inner.lock();
        let entry = g
            .tensors
            .get(tensor.as_str())
            .ok_or_else(|| MemoryError::NotResident {
                tensor: tensor.to_string(),
                resource: resource.to_string(),
            })?;
        entry
            .copies
            .get(resource.as_str())
            .filter(|c| c.valid)
            .cloned()
            .ok_or_else(|| MemoryError::NotResident {
                tensor: tensor.to_string(),
                resource: resource.to_string(),
            })
    }

    pub fn acquire_lease(&self, tensor: &TensorId, resource: &ResourceId) -> MemoryResult<()> {
        let mut g = self.inner.lock();
        let copy = g
            .tensors
            .get_mut(tensor.as_str())
            .and_then(|t| t.copies.get_mut(resource.as_str()))
            .ok_or_else(|| MemoryError::NotResident {
                tensor: tensor.to_string(),
                resource: resource.to_string(),
            })?;
        if !copy.valid {
            return Err(MemoryError::NotResident {
                tensor: tensor.to_string(),
                resource: resource.to_string(),
            });
        }
        copy.active_leases = copy.active_leases.saturating_add(1);
        Ok(())
    }

    pub fn release_lease(&self, tensor: &TensorId, resource: &ResourceId) -> MemoryResult<()> {
        let mut g = self.inner.lock();
        let copy = g
            .tensors
            .get_mut(tensor.as_str())
            .and_then(|t| t.copies.get_mut(resource.as_str()))
            .ok_or_else(|| MemoryError::NotResident {
                tensor: tensor.to_string(),
                resource: resource.to_string(),
            })?;
        if copy.active_leases == 0 {
            return Err(MemoryError::LeaseUnderflow {
                tensor: tensor.to_string(),
                resource: resource.to_string(),
            });
        }
        copy.active_leases -= 1;
        Ok(())
    }

    /// Release exact-resource copy. Fails if leases active. Frees allocation on final ref.
    pub fn release_copy(&self, tensor: &TensorId, resource: &ResourceId) -> MemoryResult<u64> {
        let mut g = self.inner.lock();
        let entry = g
            .tensors
            .get_mut(tensor.as_str())
            .ok_or_else(|| MemoryError::NotResident {
                tensor: tensor.to_string(),
                resource: resource.to_string(),
            })?;
        let copy = entry
            .copies
            .get(resource.as_str())
            .ok_or_else(|| MemoryError::NotResident {
                tensor: tensor.to_string(),
                resource: resource.to_string(),
            })?;
        if copy.active_leases > 0 {
            return Err(MemoryError::AliasUnsafeRelease {
                tensor: tensor.to_string(),
                leases: copy.active_leases,
            });
        }
        let alloc = copy.allocation.clone();
        entry.copies.remove(resource.as_str());
        drop(g);
        self.allocations.release(&alloc)
    }

    /// Deduplicate in-progress transfers to the same (tensor, dest).
    pub fn begin_transfer(
        &self,
        tensor: &TensorId,
        dest: &ResourceId,
        transfer_id: impl Into<String>,
    ) -> Option<String> {
        let mut g = self.inner.lock();
        let key = (tensor.as_str().to_owned(), dest.as_str().to_owned());
        if let Some(existing) = g.inflight_transfers.get(&key) {
            return Some(existing.clone());
        }
        g.inflight_transfers.insert(key, transfer_id.into());
        None
    }

    pub fn end_transfer(&self, tensor: &TensorId, dest: &ResourceId) {
        let mut g = self.inner.lock();
        g.inflight_transfers
            .remove(&(tensor.as_str().to_owned(), dest.as_str().to_owned()));
    }

    #[must_use]
    pub fn logical_version(&self, tensor: &TensorId) -> u64 {
        self.inner
            .lock()
            .tensors
            .get(tensor.as_str())
            .map(|t| t.version)
            .unwrap_or(0)
    }

    #[must_use]
    pub fn snapshot_copies(&self) -> HashMap<String, Vec<String>> {
        let g = self.inner.lock();
        g.tensors
            .iter()
            .map(|(tid, rec)| {
                let resources: Vec<String> = rec
                    .copies
                    .iter()
                    .filter(|(_, c)| c.valid)
                    .map(|(r, _)| r.clone())
                    .collect();
                (tid.clone(), resources)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn put_invalidates_sibling() {
        let allocs = Arc::new(AllocationTable::new());
        let store = ResidencyStore::new(allocs);
        let meta = TensorMetadata {
            nbytes: 64,
            dtype: "float32".into(),
            ..Default::default()
        };
        store
            .put(
                TensorId::new("t"),
                ResourceId::new("cpu"),
                AllocationId::new("a1"),
                meta.clone(),
                None,
            )
            .unwrap();
        store
            .put(
                TensorId::new("t"),
                ResourceId::new("mock0"),
                AllocationId::new("a2"),
                meta,
                None,
            )
            .unwrap();
        assert!(store
            .get(&TensorId::new("t"), &ResourceId::new("cpu"))
            .is_err());
        assert!(store
            .get(&TensorId::new("t"), &ResourceId::new("mock0"))
            .is_ok());
        assert_eq!(store.logical_version(&TensorId::new("t")), 2);
    }

    #[test]
    fn replicate_keeps_both_valid() {
        let allocs = Arc::new(AllocationTable::new());
        let store = ResidencyStore::new(allocs);
        let meta = TensorMetadata {
            nbytes: 32,
            dtype: "float32".into(),
            ..Default::default()
        };
        store
            .put(
                TensorId::new("t"),
                ResourceId::new("cpu"),
                AllocationId::new("a1"),
                meta,
                None,
            )
            .unwrap();
        store
            .replicate(
                &TensorId::new("t"),
                ResourceId::new("mock0"),
                AllocationId::new("a2"),
                None,
            )
            .unwrap();
        assert!(store
            .get(&TensorId::new("t"), &ResourceId::new("cpu"))
            .is_ok());
        assert!(store
            .get(&TensorId::new("t"), &ResourceId::new("mock0"))
            .is_ok());
        assert_eq!(store.logical_version(&TensorId::new("t")), 1);
        assert_eq!(store.allocations().live_bytes(), 64);
    }

    #[test]
    fn lease_blocks_release() {
        let allocs = Arc::new(AllocationTable::new());
        let store = ResidencyStore::new(allocs);
        let meta = TensorMetadata {
            nbytes: 8,
            ..Default::default()
        };
        store
            .put(
                TensorId::new("t"),
                ResourceId::new("cpu"),
                AllocationId::new("a"),
                meta,
                None,
            )
            .unwrap();
        store
            .acquire_lease(&TensorId::new("t"), &ResourceId::new("cpu"))
            .unwrap();
        assert!(store
            .release_copy(&TensorId::new("t"), &ResourceId::new("cpu"))
            .is_err());
        store
            .release_lease(&TensorId::new("t"), &ResourceId::new("cpu"))
            .unwrap();
        assert_eq!(
            store
                .release_copy(&TensorId::new("t"), &ResourceId::new("cpu"))
                .unwrap(),
            8
        );
    }

    #[test]
    fn transfer_dedup() {
        let store = ResidencyStore::new(Arc::new(AllocationTable::new()));
        assert!(store
            .begin_transfer(&TensorId::new("t"), &ResourceId::new("gpu"), "xfer1")
            .is_none());
        assert_eq!(
            store.begin_transfer(&TensorId::new("t"), &ResourceId::new("gpu"), "xfer2"),
            Some("xfer1".into())
        );
        store.end_transfer(&TensorId::new("t"), &ResourceId::new("gpu"));
        assert!(store
            .begin_transfer(&TensorId::new("t"), &ResourceId::new("gpu"), "xfer3")
            .is_none());
    }
}
