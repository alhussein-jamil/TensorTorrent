//! Immutable executable schedule shared by planner, simulator, and runtime.

use crate::instruction::Instruction;
use crate::opcode::Opcode;
use serde::{Deserialize, Serialize};

/// Immutable executable plan: same object for plan explain, sim, and run.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExecutableSchedule {
    pub graph_name: String,
    pub fingerprint: String,
    #[serde(default)]
    pub instructions: Vec<Instruction>,
    #[serde(default)]
    pub notes: Vec<String>,
}

impl ExecutableSchedule {
    #[must_use]
    pub fn new(
        graph_name: impl Into<String>,
        fingerprint: impl Into<String>,
        instructions: Vec<Instruction>,
        notes: Vec<String>,
    ) -> Self {
        let mut schedule = Self {
            graph_name: graph_name.into(),
            fingerprint: fingerprint.into(),
            instructions,
            notes,
        };
        schedule.ensure_explicit_streams();
        schedule
    }

    /// Fill missing stream / copy-engine / link ids with deterministic defaults.
    pub fn ensure_explicit_streams(&mut self) {
        use crate::ids::StreamId;
        use crate::opcode::Opcode;
        for inst in &mut self.instructions {
            let resource = inst.resource.as_str();
            if inst
                .stream_id
                .as_ref()
                .map(|s| s.as_str().is_empty())
                .unwrap_or(true)
            {
                let sid = match inst.opcode {
                    Opcode::Compute => format!("{resource}::compute"),
                    Opcode::Transfer | Opcode::Load | Opcode::Prefetch => {
                        format!("{resource}::copy0")
                    }
                    Opcode::RecordEvent | Opcode::WaitEvent => format!("{resource}::sync"),
                    Opcode::Evict | Opcode::Release => format!("{resource}::lifetime"),
                };
                inst.stream_id = Some(StreamId::new(sid));
            }
            if matches!(
                inst.opcode,
                Opcode::Transfer | Opcode::Load | Opcode::Prefetch
            ) && inst
                .copy_engine_id
                .as_ref()
                .map(|s| s.is_empty())
                .unwrap_or(true)
            {
                inst.copy_engine_id = Some(format!("{resource}::copy0"));
            }
            if inst.opcode == Opcode::Transfer
                && inst.link_id.as_ref().map(|s| s.is_empty()).unwrap_or(true)
            {
                let src = inst
                    .source
                    .as_ref()
                    .map(|s| s.as_str())
                    .unwrap_or("unknown");
                let dst = inst
                    .destination
                    .as_ref()
                    .map(|s| s.as_str())
                    .unwrap_or(resource);
                inst.link_id = Some(format!("{src}->{dst}"));
            }
        }
    }

    #[must_use]
    pub fn compute_ops(&self) -> Vec<&Instruction> {
        self.instructions
            .iter()
            .filter(|i| i.opcode == Opcode::Compute)
            .collect()
    }

    #[must_use]
    pub fn transfer_ops(&self) -> Vec<&Instruction> {
        self.instructions
            .iter()
            .filter(|i| matches!(i.opcode, Opcode::Transfer | Opcode::Prefetch | Opcode::Load))
            .collect()
    }

    /// Serialize to JSON (lossless for the Rust model).
    pub fn to_json(&self) -> Result<String, crate::CoreError> {
        serde_json::to_string(self).map_err(|e| crate::CoreError::Serde(e.to_string()))
    }

    /// Deserialize from JSON.
    pub fn from_json(s: &str) -> Result<Self, crate::CoreError> {
        let mut schedule: Self =
            serde_json::from_str(s).map_err(|e| crate::CoreError::Serde(e.to_string()))?;
        schedule.ensure_explicit_streams();
        Ok(schedule)
    }

    /// Compact binary-safe JSON bytes.
    pub fn to_json_bytes(&self) -> Result<Vec<u8>, crate::CoreError> {
        serde_json::to_vec(self).map_err(|e| crate::CoreError::Serde(e.to_string()))
    }

    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, crate::CoreError> {
        let mut schedule: Self =
            serde_json::from_slice(bytes).map_err(|e| crate::CoreError::Serde(e.to_string()))?;
        schedule.ensure_explicit_streams();
        Ok(schedule)
    }
}
