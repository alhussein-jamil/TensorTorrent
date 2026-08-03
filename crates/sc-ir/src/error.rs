//! Error types for the schedule core.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum CoreError {
    #[error("schedule validation failed: {0}")]
    Validation(String),
    #[error("serialization error: {0}")]
    Serde(String),
    #[error("unknown opcode: {0}")]
    UnknownOpcode(String),
    #[error("malformed attribute for instruction {instruction}: {detail}")]
    BadAttribute { instruction: String, detail: String },
    #[error(
        "incomplete schedule: instruction {instruction} opcode {opcode} missing exact tensor sizes"
    )]
    IncompleteSizes { instruction: String, opcode: String },
}

pub type CoreResult<T> = Result<T, CoreError>;
