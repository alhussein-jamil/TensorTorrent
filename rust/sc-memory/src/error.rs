use thiserror::Error;

#[derive(Debug, Error)]
pub enum MemoryError {
    #[error("tensor {tensor} not resident on resource {resource}")]
    NotResident { tensor: String, resource: String },
    #[error("tensor {tensor} version mismatch: expected {expected}, found {found}")]
    VersionMismatch {
        tensor: String,
        expected: u64,
        found: u64,
    },
    #[error("allocation {0} not found")]
    AllocationMissing(String),
    #[error("lease underflow for tensor {tensor} on {resource}")]
    LeaseUnderflow { tensor: String, resource: String },
    #[error("alias-unsafe release of tensor {tensor}: active_leases={leases}")]
    AliasUnsafeRelease { tensor: String, leases: u32 },
    #[error("capacity exceeded on resource {resource}: need {need}, free {free}")]
    CapacityExceeded {
        resource: String,
        need: u64,
        free: u64,
    },
    #[error("{0}")]
    Other(String),
}

pub type MemoryResult<T> = Result<T, MemoryError>;
