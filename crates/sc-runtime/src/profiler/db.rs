use crate::profiler::record::ProfileRecord;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

#[derive(Debug, Default)]
pub struct ProfileDatabase {
    records: Mutex<HashMap<String, Vec<ProfileRecord>>>,
}

impl ProfileDatabase {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&self, record: ProfileRecord) {
        self.records
            .lock()
            .entry(record.cache_key.clone())
            .or_default()
            .push(record);
    }

    #[must_use]
    pub fn get(&self, cache_key: &str) -> Vec<ProfileRecord> {
        self.records
            .lock()
            .get(cache_key)
            .cloned()
            .unwrap_or_default()
    }

    pub fn aggregate_median_latency(&self, cache_key: &str) -> Option<f64> {
        let recs = self.get(cache_key);
        let mut all = Vec::new();
        for r in &recs {
            all.extend(r.costs.iter().cloned());
        }
        ProfileRecord::median_latency(&all)
    }

    pub fn save_json(&self, path: impl AsRef<Path>) -> Result<(), String> {
        let g = self.records.lock();
        let s = serde_json::to_string_pretty(&*g).map_err(|e| e.to_string())?;
        fs::write(path, s).map_err(|e| e.to_string())
    }

    pub fn load_json(&self, path: impl AsRef<Path>) -> Result<(), String> {
        let s = fs::read_to_string(path).map_err(|e| e.to_string())?;
        let map: HashMap<String, Vec<ProfileRecord>> =
            serde_json::from_str(&s).map_err(|e| e.to_string())?;
        *self.records.lock() = map;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profiler::record::{CostStatus, RegionCost};

    #[test]
    fn persist_roundtrip() {
        let db = ProfileDatabase::new();
        db.insert(ProfileRecord {
            cache_key: "k".into(),
            costs: vec![RegionCost {
                region_id: "r".into(),
                resource: "cpu".into(),
                latency_s: 0.01,
                workspace_bytes: 0,
                status: CostStatus::Measured,
            }],
            transfer_latency_s: 0.0,
            io_latency_s: 0.0,
            status: CostStatus::Measured,
            notes: vec![],
        });
        let path = std::env::temp_dir().join("sc-profile-test.json");
        db.save_json(&path).unwrap();
        let db2 = ProfileDatabase::new();
        db2.load_json(&path).unwrap();
        assert_eq!(db2.get("k").len(), 1);
        assert!((db2.aggregate_median_latency("k").unwrap() - 0.01).abs() < 1e-12);
    }
}
