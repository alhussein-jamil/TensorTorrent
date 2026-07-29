use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InstructionTelemetry {
    pub name: String,
    pub opcode: String,
    pub resource: String,
    pub submitted_s: f64,
    pub start_s: f64,
    pub end_s: f64,
    pub nbytes: u64,
    pub simulated: bool,
    pub notes: String,
}
