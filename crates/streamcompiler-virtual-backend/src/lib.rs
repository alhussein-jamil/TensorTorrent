//! Deterministic virtual accelerator.
//!
//! All buffers are distinct from host tensors. All timings are simulated.
//! Never claim real CUDA/ROCm behavior.

mod backend;

pub use backend::{VirtualBackend, VirtualBackendConfig};
