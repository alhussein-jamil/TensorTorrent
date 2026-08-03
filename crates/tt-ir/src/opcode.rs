//! Schedule opcodes matching the Python `OpCode` values used by the runtime path.

use crate::error::{CoreError, CoreResult};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

/// Runtime-executable opcodes. Values match Python `OpCode.value` strings.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub enum Opcode {
    Prefetch,
    Load,
    Transfer,
    RecordEvent,
    WaitEvent,
    Compute,
    Evict,
    Release,
}

impl Opcode {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Prefetch => "Prefetch",
            Self::Load => "Load",
            Self::Transfer => "Transfer",
            Self::RecordEvent => "RecordEvent",
            Self::WaitEvent => "WaitEvent",
            Self::Compute => "Compute",
            Self::Evict => "Evict",
            Self::Release => "Release",
        }
    }

    #[must_use]
    pub fn is_runtime_supported(self) -> bool {
        true
    }
}

impl fmt::Display for Opcode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for Opcode {
    type Err = CoreError;

    fn from_str(s: &str) -> CoreResult<Self> {
        match s {
            "Prefetch" => Ok(Self::Prefetch),
            "Load" => Ok(Self::Load),
            "Transfer" => Ok(Self::Transfer),
            "RecordEvent" => Ok(Self::RecordEvent),
            "WaitEvent" => Ok(Self::WaitEvent),
            "Compute" => Ok(Self::Compute),
            "Evict" => Ok(Self::Evict),
            "Release" => Ok(Self::Release),
            other => Err(CoreError::UnknownOpcode(other.to_owned())),
        }
    }
}
