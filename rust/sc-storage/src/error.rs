use thiserror::Error;

#[derive(Debug, Error)]
pub enum StorageError {
    #[error("pack I/O: {0}")]
    Io(String),
    #[error("invalid pack: {0}")]
    Invalid(String),
    #[error("checksum mismatch for tensor {tensor}: expected {expected}, got {got}")]
    ChecksumMismatch {
        tensor: String,
        expected: String,
        got: String,
    },
    #[error(
        "offset overflow or out of bounds: offset={offset} length={length} file_size={file_size}"
    )]
    Bounds {
        offset: u64,
        length: u64,
        file_size: u64,
    },
    #[error("excessive allocation request: {0} bytes")]
    ExcessiveAllocation(u64),
}

pub type StorageResult<T> = Result<T, StorageError>;
