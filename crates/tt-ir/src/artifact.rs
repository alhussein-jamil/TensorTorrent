//! Versioned immutable executable artifact produced by compilation.

use crate::error::{CoreError, CoreResult};
use crate::schedule::ExecutableSchedule;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Compatibility version for on-disk / in-memory artifacts.
pub const ARTIFACT_FORMAT_VERSION: u32 = 1;

/// Exact tensor metadata stored with the artifact.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactTensorMeta {
    pub tensor_id: String,
    pub dtype: String,
    pub shape: Vec<i64>,
    pub strides: Vec<i64>,
    pub nbytes: u64,
    pub storage_key: Option<String>,
}

/// Initial residency for persistent parameters already loaded with the artifact.
/// These are NOT fake runtime `Load` instructions.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct InitialResidency {
    pub tensor_id: String,
    pub resource_id: String,
    pub memory_domain_id: String,
    pub version: u64,
    pub nbytes: u64,
}

/// Resource requirement declared by the compiled plan.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceRequirement {
    pub resource_id: String,
    pub kind: String,
    pub min_memory_bytes: u64,
}

/// Storage manifest reference (pack path + checksum).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StorageManifestRef {
    pub pack_path: String,
    pub checksum: Option<String>,
    pub format: String,
}

/// Memory plan summary (budgets and domains).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryPlan {
    pub host_budget_bytes: u64,
    pub device_budget_bytes: u64,
    pub activation_budget_bytes: Option<u64>,
    pub domains: BTreeMap<String, u64>,
}

/// Immutable, versioned executable artifact.
///
/// Compilation produces this once. Forward execution must not rebuild or
/// reinterpret the graph. Serialization uses JSON (never pickle).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExecutableArtifact {
    pub format_version: u32,
    pub compatibility_version: String,
    pub graph_identity: String,
    pub schedule: ExecutableSchedule,
    pub tensors: Vec<ArtifactTensorMeta>,
    pub resource_requirements: Vec<ResourceRequirement>,
    pub storage_manifest: Option<StorageManifestRef>,
    pub memory_plan: MemoryPlan,
    pub initial_residency: Vec<InitialResidency>,
    pub profile_keys: Vec<String>,
    pub compiled_region_ids: Vec<String>,
    pub notes: Vec<String>,
}

impl ExecutableArtifact {
    #[must_use]
    pub fn from_schedule(schedule: ExecutableSchedule) -> Self {
        let graph_identity = format!("{}::{}", schedule.graph_name, schedule.fingerprint);
        Self {
            format_version: ARTIFACT_FORMAT_VERSION,
            compatibility_version: format!("tt-artifact-v{ARTIFACT_FORMAT_VERSION}"),
            graph_identity,
            schedule,
            tensors: Vec::new(),
            resource_requirements: Vec::new(),
            storage_manifest: None,
            memory_plan: MemoryPlan {
                host_budget_bytes: 0,
                device_budget_bytes: 0,
                activation_budget_bytes: None,
                domains: BTreeMap::new(),
            },
            initial_residency: Vec::new(),
            profile_keys: Vec::new(),
            compiled_region_ids: Vec::new(),
            notes: Vec::new(),
        }
    }

    pub fn validate(&self) -> CoreResult<()> {
        if self.format_version != ARTIFACT_FORMAT_VERSION {
            return Err(CoreError::Validation(format!(
                "unsupported artifact format_version {} (want {ARTIFACT_FORMAT_VERSION})",
                self.format_version
            )));
        }
        if self.graph_identity.is_empty() {
            return Err(CoreError::Validation(
                "artifact graph_identity must be non-empty".into(),
            ));
        }
        crate::validate::assert_schedule_valid(&self.schedule)?;
        for t in &self.tensors {
            if t.nbytes == 0 && !t.shape.is_empty() {
                return Err(CoreError::Validation(format!(
                    "tensor {} has empty nbytes with non-empty shape",
                    t.tensor_id
                )));
            }
        }
        for r in &self.initial_residency {
            if r.resource_id.is_empty() || r.memory_domain_id.is_empty() {
                return Err(CoreError::Validation(format!(
                    "initial residency for {} missing resource/domain",
                    r.tensor_id
                )));
            }
        }
        Ok(())
    }

    pub fn to_json_bytes(&self) -> CoreResult<Vec<u8>> {
        serde_json::to_vec(self).map_err(|e| CoreError::Serde(e.to_string()))
    }

    pub fn from_json_bytes(bytes: &[u8]) -> CoreResult<Self> {
        let art: Self =
            serde_json::from_slice(bytes).map_err(|e| CoreError::Serde(e.to_string()))?;
        art.validate()?;
        Ok(art)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ExecutableSchedule;

    #[test]
    fn artifact_roundtrip_json() {
        let schedule = ExecutableSchedule::new("g", "fp", vec![], vec![]);
        let art = ExecutableArtifact::from_schedule(schedule);
        art.validate().unwrap();
        let bytes = art.to_json_bytes().unwrap();
        let loaded = ExecutableArtifact::from_json_bytes(&bytes).unwrap();
        assert_eq!(loaded.format_version, ARTIFACT_FORMAT_VERSION);
        assert_eq!(loaded.schedule.graph_name, "g");
    }
}
