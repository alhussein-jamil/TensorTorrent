//! Immutable instructions and attribute values.

use crate::ids::{InstructionId, RegionId, ResourceId, TensorId};
use crate::opcode::Opcode;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Memory tier classification matching Python `MemoryTier`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum MemoryTier {
    Disk,
    SystemRam,
    PinnedRam,
    NumaRam,
    Device,
    #[default]
    Unknown,
}

impl MemoryTier {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Disk => "disk",
            Self::SystemRam => "system_ram",
            Self::PinnedRam => "pinned_ram",
            Self::NumaRam => "numa_ram",
            Self::Device => "device",
            Self::Unknown => "unknown",
        }
    }
}

impl std::str::FromStr for MemoryTier {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "disk" => Ok(Self::Disk),
            "system_ram" => Ok(Self::SystemRam),
            "pinned_ram" => Ok(Self::PinnedRam),
            "numa_ram" => Ok(Self::NumaRam),
            "device" => Ok(Self::Device),
            "unknown" => Ok(Self::Unknown),
            other => Err(format!("unknown memory tier: {other}")),
        }
    }
}

/// JSON-compatible attribute value for lossless Python round-trips.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AttrValue {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
    IntMap(BTreeMap<String, i64>),
    StringMap(BTreeMap<String, String>),
    List(Vec<AttrValue>),
    Map(BTreeMap<String, AttrValue>),
}

impl AttrValue {
    #[must_use]
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s),
            _ => None,
        }
    }

    #[must_use]
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Self::Int(v) => Some(*v),
            Self::Float(v) => Some(*v as i64),
            _ => None,
        }
    }

    #[must_use]
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(v) => Some(*v),
            _ => None,
        }
    }

    #[must_use]
    pub fn as_int_map(&self) -> Option<&BTreeMap<String, i64>> {
        match self {
            Self::IntMap(m) => Some(m),
            Self::Map(m) => {
                // Accept Map of ints via conversion check elsewhere.
                let _ = m;
                None
            }
            _ => None,
        }
    }
}

/// One immutable scheduled op shared by planner, simulator, and runtime.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Instruction {
    pub opcode: Opcode,
    pub name: InstructionId,
    pub resource: ResourceId,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub depends_on: Vec<InstructionId>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub inputs: Vec<TensorId>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub outputs: Vec<TensorId>,
    #[serde(default)]
    pub nbytes: u64,
    #[serde(default)]
    pub memory_tier: MemoryTier,
    #[serde(default)]
    pub predicted_duration_s: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub executable_ref: Option<RegionId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<ResourceId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub destination: Option<ResourceId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transfer_backend: Option<String>,
    #[serde(default)]
    pub sync_required: bool,
    /// Sorted attribute map for deterministic serialization.
    #[serde(default, skip_serializing_if = "IndexMap::is_empty")]
    pub attributes: IndexMap<String, AttrValue>,
}

impl Instruction {
    /// Exact per-tensor sizes from attributes, or empty if absent.
    #[must_use]
    pub fn tensor_nbytes(&self) -> BTreeMap<String, u64> {
        let raw = self
            .attributes
            .get("tensor_nbytes")
            .or_else(|| self.attributes.get("output_bytes"));
        match raw {
            Some(AttrValue::IntMap(m)) => m
                .iter()
                .map(|(k, v)| (k.clone(), (*v).max(0) as u64))
                .collect(),
            Some(AttrValue::Map(m)) => {
                let mut out = BTreeMap::new();
                for (k, v) in m {
                    if let Some(n) = v.as_i64() {
                        out.insert(k.clone(), n.max(0) as u64);
                    }
                }
                out
            }
            _ => BTreeMap::new(),
        }
    }

    #[must_use]
    pub fn attr_str(&self, key: &str) -> Option<&str> {
        self.attributes.get(key).and_then(AttrValue::as_str)
    }

    #[must_use]
    pub fn attr_bool(&self, key: &str) -> Option<bool> {
        self.attributes.get(key).and_then(AttrValue::as_bool)
    }
}
