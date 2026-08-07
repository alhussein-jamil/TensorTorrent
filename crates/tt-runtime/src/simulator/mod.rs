//! Deterministic discrete-event simulator for executable schedules.
//!
//! Analytic only: kernels are not executed. All outputs are labelled `simulated=true`.
//! Does not invent data movements absent from the schedule.

mod machine;
mod sim;

pub use machine::{link_class_prior, MachineModel, MemoryResource, TransferEstimate, TransferLink};
pub use sim::{
    should_parallelize_batch_des, simulate_schedule, simulate_schedules,
    simulate_schedules_with_stats, BatchSimStatistics, InfeasibilityReport, SimulationOutcome,
    SimulationResult, TimelineEvent,
};
