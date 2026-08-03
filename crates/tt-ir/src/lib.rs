//! TensorTorrent IR: typed IDs, opcodes, immutable schedules, artifacts, validation.

pub mod artifact;
pub mod error;
pub mod ids;
pub mod instruction;
pub mod opcode;
pub mod schedule;
pub mod validate;

pub use artifact::{
    ArtifactTensorMeta, ExecutableArtifact, InitialResidency, MemoryPlan, ResourceRequirement,
    StorageManifestRef, ARTIFACT_FORMAT_VERSION,
};
pub use error::{CoreError, CoreResult};
pub use ids::{AllocationId, EventId, InstructionId, RegionId, ResourceId, StreamId, TensorId};
pub use instruction::{AttrValue, Instruction, MemoryTier};
pub use opcode::Opcode;
pub use schedule::ExecutableSchedule;
pub use validate::{
    assert_schedule_valid, validate_schedule, validate_tensor_sizes, ValidationReport,
};
