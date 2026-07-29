//! Deterministic discrete-event simulator for executable schedules.
//!
//! Analytic only: kernels are not executed. All outputs are labelled `simulated=true`.
//! Does not invent data movements absent from the schedule.

mod machine;
mod sim;

pub use machine::{MachineModel, MemoryResource, TransferLink};
pub use sim::{simulate_schedule, SimulationResult, TimelineEvent};
