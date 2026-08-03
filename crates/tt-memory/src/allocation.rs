//! Physical allocation accounting: views share one allocation; distinct resources count separately.

use crate::error::{MemoryError, MemoryResult};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tt_ir::{AllocationId, ResourceId};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PhysicalAllocation {
    pub id: AllocationId,
    pub resource: ResourceId,
    pub capacity_bytes: u64,
    pub used_bytes: u64,
    pub alignment: u64,
    pub reference_count: u32,
}

#[derive(Debug, Default)]
pub struct AllocationTable {
    inner: Mutex<Inner>,
}

#[derive(Debug, Default)]
struct Inner {
    allocs: HashMap<String, PhysicalAllocation>,
    live_bytes: u64,
    peak_bytes: u64,
    /// Optional per-resource capacity ceilings (None = unbounded).
    capacity_limits: HashMap<String, u64>,
    live_by_resource: HashMap<String, u64>,
}

impl AllocationTable {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set_capacity_limit(&self, resource: impl Into<String>, bytes: u64) {
        self.inner
            .lock()
            .capacity_limits
            .insert(resource.into(), bytes);
    }

    /// Register or bump reference count. Same allocation id on same resource is shared.
    pub fn register(
        &self,
        id: AllocationId,
        resource: ResourceId,
        capacity_bytes: u64,
        alignment: u64,
    ) -> MemoryResult<()> {
        let mut g = self.inner.lock();
        let key = id.as_str().to_owned();
        if let Some(rec) = g.allocs.get_mut(&key) {
            if rec.capacity_bytes != capacity_bytes || rec.alignment != alignment {
                return Err(MemoryError::Other(format!(
                    "allocation {key} metadata mismatch: capacity/alignment {}/{} != {capacity_bytes}/{alignment}",
                    rec.capacity_bytes, rec.alignment
                )));
            }
            rec.reference_count = rec.reference_count.checked_add(1).ok_or_else(|| {
                MemoryError::Other(format!("allocation {key} reference count overflow"))
            })?;
            return Ok(());
        }
        let res_key = resource.as_str().to_owned();
        let live = g.live_by_resource.get(&res_key).copied().unwrap_or(0);
        if let Some(limit) = g.capacity_limits.get(&res_key).copied() {
            if live.saturating_add(capacity_bytes) > limit {
                return Err(MemoryError::CapacityExceeded {
                    resource: res_key,
                    need: capacity_bytes,
                    free: limit.saturating_sub(live),
                });
            }
        }
        g.allocs.insert(
            key,
            PhysicalAllocation {
                id,
                resource,
                capacity_bytes,
                used_bytes: capacity_bytes,
                alignment,
                reference_count: 1,
            },
        );
        g.live_bytes = g.live_bytes.saturating_add(capacity_bytes);
        g.peak_bytes = g.peak_bytes.max(g.live_bytes);
        *g.live_by_resource.entry(res_key).or_insert(0) = live.saturating_add(capacity_bytes);
        Ok(())
    }

    /// Drop one reference; free capacity on final reference.
    pub fn release(&self, id: &AllocationId) -> MemoryResult<u64> {
        let mut g = self.inner.lock();
        let key = id.as_str();
        let Some(rec) = g.allocs.get_mut(key) else {
            return Err(MemoryError::AllocationMissing(key.to_owned()));
        };
        if rec.reference_count == 0 {
            return Err(MemoryError::AllocationMissing(key.to_owned()));
        }
        rec.reference_count -= 1;
        if rec.reference_count > 0 {
            return Ok(0);
        }
        let freed = rec.capacity_bytes;
        let res_key = rec.resource.as_str().to_owned();
        g.allocs.remove(key);
        g.live_bytes = g.live_bytes.saturating_sub(freed);
        if let Some(v) = g.live_by_resource.get_mut(&res_key) {
            *v = v.saturating_sub(freed);
        }
        Ok(freed)
    }

    #[must_use]
    pub fn live_bytes(&self) -> u64 {
        self.inner.lock().live_bytes
    }

    #[must_use]
    pub fn peak_bytes(&self) -> u64 {
        self.inner.lock().peak_bytes
    }

    #[must_use]
    pub fn live_bytes_by_resource(&self) -> HashMap<String, u64> {
        self.inner.lock().live_by_resource.clone()
    }

    #[must_use]
    pub fn reference_count(&self, id: &AllocationId) -> Option<u32> {
        self.inner
            .lock()
            .allocs
            .get(id.as_str())
            .map(|a| a.reference_count)
    }

    #[must_use]
    pub fn get(&self, id: &AllocationId) -> Option<PhysicalAllocation> {
        self.inner.lock().allocs.get(id.as_str()).cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tt_ir::{AllocationId, ResourceId};

    #[test]
    fn shared_views_one_allocation() {
        let table = AllocationTable::new();
        let id = AllocationId::new("stor0");
        table
            .register(id.clone(), ResourceId::new("cpu"), 1024, 64)
            .unwrap();
        table
            .register(id.clone(), ResourceId::new("cpu"), 1024, 64)
            .unwrap();
        assert_eq!(table.live_bytes(), 1024);
        assert_eq!(table.reference_count(&id), Some(2));
        assert_eq!(table.release(&id).unwrap(), 0);
        assert_eq!(table.live_bytes(), 1024);
        assert_eq!(table.release(&id).unwrap(), 1024);
        assert_eq!(table.live_bytes(), 0);
    }

    #[test]
    fn distinct_resources_count_separately() {
        let table = AllocationTable::new();
        table
            .register(AllocationId::new("a"), ResourceId::new("cpu"), 100, 8)
            .unwrap();
        table
            .register(AllocationId::new("b"), ResourceId::new("mock0"), 100, 8)
            .unwrap();
        assert_eq!(table.live_bytes(), 200);
        let by = table.live_bytes_by_resource();
        assert_eq!(by.get("cpu"), Some(&100));
        assert_eq!(by.get("mock0"), Some(&100));
    }

    #[test]
    fn shared_allocation_rejects_conflicting_metadata() {
        let table = AllocationTable::new();
        let id = AllocationId::new("shared");
        table
            .register(id.clone(), ResourceId::new("cpu"), 100, 8)
            .unwrap();
        assert!(table
            .register(id.clone(), ResourceId::new("host"), 101, 8)
            .is_err());
        assert_eq!(table.reference_count(&id), Some(1));
        assert_eq!(table.live_bytes(), 100);
    }

    #[test]
    fn capacity_limit_refuses_overflow_and_frees_recover() {
        let table = AllocationTable::new();
        table.set_capacity_limit("numa_ram_0", 1024);
        table
            .register(
                AllocationId::new("a1"),
                ResourceId::new("numa_ram_0"),
                800,
                64,
            )
            .unwrap();
        let err = table
            .register(
                AllocationId::new("a2"),
                ResourceId::new("numa_ram_0"),
                512,
                64,
            )
            .unwrap_err();
        match err {
            MemoryError::CapacityExceeded {
                resource,
                need,
                free,
            } => {
                assert_eq!(resource, "numa_ram_0");
                assert_eq!(need, 512);
                assert_eq!(free, 224);
            }
            other => panic!("expected CapacityExceeded, got {other:?}"),
        }
        // Releasing the first allocation must make room again.
        table.release(&AllocationId::new("a1")).unwrap();
        table
            .register(
                AllocationId::new("a2"),
                ResourceId::new("numa_ram_0"),
                512,
                64,
            )
            .unwrap();
    }
}
