//! Event-driven schedule runtime. Releases no Python GIL itself — callers hold none.

mod error;
mod executor;
mod telemetry;
mod workers;

pub use error::{RuntimeError, RuntimeResult};
pub use executor::{
    execute_schedule, execute_schedule_ex, ExecuteOptions, ExecuteReport, InstructionCallback,
    InstructionCallbackResult, RegionCallback,
};
pub use telemetry::InstructionTelemetry;
pub use workers::WorkerPool;
