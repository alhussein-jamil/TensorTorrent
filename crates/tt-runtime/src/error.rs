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
    #[error(
        "stalled: no progress for {waited_s:.1}s while waiting for {what}. This usually \
         means a lost completion or a deadlocked resource; if this host legitimately has \
         I/O this slow, raise CompileConfig.stall_timeout_s"
    )]
    Stalled { what: String, waited_s: f64 },
    #[error("{0}")]
    Other(String),
}

// Clippy result_large_err: instruction context strings dominate; boxing keeps Result small.
pub type RuntimeResult<T> = Result<T, Box<RuntimeError>>;

impl From<tt_ir::CoreError> for Box<RuntimeError> {
    fn from(e: tt_ir::CoreError) -> Self {
        Box::new(RuntimeError::Validation(e.to_string()))
    }
}
