//! Profile records with explicit cost provenance.

mod db;
mod record;

pub use db::ProfileDatabase;
pub use record::{CostStatus, ProfileRecord, RegionCost};
