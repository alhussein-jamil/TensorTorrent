//! Pack format parsing and safe positional I/O.
//!
//! Never deserializes executable code from packs. All metadata is untrusted.

mod cache;
mod error;
mod pack;
mod streaming;

pub use cache::ChunkCache;
pub use error::{StorageError, StorageResult};
pub use pack::{PackManifest, PackReader, TensorEntry, PACK_FORMAT_VERSION};
pub use streaming::{StreamingStats, StreamingStore};
