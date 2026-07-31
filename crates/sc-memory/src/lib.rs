//! Authoritative residency and physical allocation model.

mod allocation;
mod error;
mod residency;

pub use allocation::{AllocationTable, PhysicalAllocation};
pub use error::{MemoryError, MemoryResult};
pub use residency::{LogicalTensorRecord, ResidencyStore, ResidentCopy, TensorMetadata};
