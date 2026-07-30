//! Event-driven schedule runtime. Releases no Python GIL itself — callers hold none.

mod context;
mod error;
mod executor;
pub mod profiler;
mod resources;
pub mod simulator;
mod telemetry;
mod workers;

pub use context::{ExecutionId, ExecutionStorageState, NativeExecutionContext};
pub use error::{RuntimeError, RuntimeResult};
pub use executor::{
    execute_schedule, execute_schedule_ex, execute_schedule_with_context, CopySyncCallback,
    DematerializeCallback, ExecuteOptions, ExecuteReport, HandleReleaseCallback,
    InstructionCallback, InstructionCallbackResult, MaterializeCallback, ParameterLoadCallback,
    RegionCallback, RegionInvocation,
};
pub use profiler::{CostStatus, ProfileDatabase, ProfileRecord, RegionCost};
pub use resources::{BandwidthState, CapacityState, OrderedStreamState, ResourceState};
pub use simulator::{
    simulate_schedule, InfeasibilityReport, MachineModel, MemoryResource, SimulationOutcome,
    SimulationResult, TimelineEvent, TransferLink,
};
pub use telemetry::{max_concurrency_from_intervals, InstructionTelemetry};
pub use workers::WorkerPool;

// Re-export production CPU backend for topology-aware hosts.
pub use sc_backend_cpu::{discover_numa_topology, CpuBackend, CpuPoolConfig, NumaTopology};
