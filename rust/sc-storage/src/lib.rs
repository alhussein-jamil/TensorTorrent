//! Pack format parsing and safe positional I/O.
//!
//! Never deserializes executable code from packs. All metadata is untrusted.

mod cache;
mod error;
mod pack;
mod spill;
mod streaming;

pub use cache::ChunkCache;
pub use error::{StorageError, StorageResult};
pub use pack::{PackManifest, PackReader, TensorEntry, PACK_FORMAT_VERSION};
pub use spill::{
    read_activation_spill, remove_activation_spill, write_activation_spill, SpillMeta,
};
pub use streaming::{StreamingStats, StreamingStore};
