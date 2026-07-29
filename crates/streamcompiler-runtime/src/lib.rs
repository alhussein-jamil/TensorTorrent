//! Event-driven schedule runtime. Releases no Python GIL itself — callers hold none.

mod context;
mod error;
mod executor;
mod resources;
mod telemetry;
mod workers;

pub use context::{ExecutionId, ExecutionStorageState, NativeExecutionContext};
pub use error::{RuntimeError, RuntimeResult};
pub use executor::{
    execute_schedule, execute_schedule_ex, execute_schedule_with_context, DematerializeCallback,
    ExecuteOptions, ExecuteReport, InstructionCallback, InstructionCallbackResult,
    MaterializeCallback, RegionCallback, RegionInvocation,
};
pub use resources::{BandwidthState, CapacityState, OrderedStreamState, ResourceState};
pub use telemetry::{max_concurrency_from_intervals, InstructionTelemetry};
pub use workers::WorkerPool;
