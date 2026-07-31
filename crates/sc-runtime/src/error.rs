use thiserror::Error;

#[derive(Debug, Error)]
pub enum RuntimeError {
    #[error("schedule validation: {0}")]
    Validation(String),
    #[error(
        "instruction {instruction} opcode {opcode} region={region:?} tensor={tensor:?} resource={resource:?}: {cause}"
    )]
    Instruction {
        instruction: String,
        opcode: String,
        region: Option<String>,
        tensor: Option<String>,
        resource: Option<String>,
        cause: String,
    },
    #[error("cancelled")]
    Cancelled,
    #[error("worker panic: {0}")]
    WorkerPanic(String),
    #[error("{0}")]
    Other(String),
}

// Clippy result_large_err: instruction context strings dominate; boxing keeps Result small.
pub type RuntimeResult<T> = Result<T, Box<RuntimeError>>;

impl From<sc_ir::CoreError> for Box<RuntimeError> {
    fn from(e: sc_ir::CoreError) -> Self {
        Box::new(RuntimeError::Validation(e.to_string()))
    }
}
