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
    create_spill_session_dir, ensure_spill_dir_usable, free_space_bytes, is_ram_backed_fs,
    read_activation_spill, remove_activation_spill, remove_spill_session_dir,
    sweep_orphan_spill_sessions, write_activation_spill, SpillMeta, SPILL_FREE_SPACE_MARGIN,
    SPILL_SESSION_PREFIX,
};
pub use streaming::{StreamingStats, StreamingStore};
