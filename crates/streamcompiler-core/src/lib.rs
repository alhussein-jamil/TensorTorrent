//! StreamCompiler core: typed IDs, opcodes, immutable schedules, validation, serde.

pub mod error;
pub mod ids;
pub mod instruction;
pub mod opcode;
pub mod schedule;
pub mod validate;

pub use error::{CoreError, CoreResult};
pub use ids::{AllocationId, EventId, InstructionId, RegionId, ResourceId, StreamId, TensorId};
pub use instruction::{AttrValue, Instruction, MemoryTier};
pub use opcode::Opcode;
pub use schedule::ExecutableSchedule;
pub use validate::{
    assert_schedule_valid, validate_schedule, validate_tensor_sizes, ValidationReport,
};
