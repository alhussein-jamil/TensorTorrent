//! Stable typed identifiers. String-backed for Python/FFI round-trips.

use serde::{Deserialize, Serialize};
use std::fmt;

macro_rules! id_newtype {
    ($name:ident, $doc:literal) => {
        #[doc = $doc]
        #[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
        #[serde(transparent)]
        pub struct $name(pub String);

        impl $name {
            #[must_use]
            pub fn new(s: impl Into<String>) -> Self {
                Self(s.into())
            }

            #[must_use]
            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl From<&str> for $name {
            fn from(s: &str) -> Self {
                Self(s.to_owned())
            }
        }

        impl From<String> for $name {
            fn from(s: String) -> Self {
                Self(s)
            }
        }

        impl AsRef<str> for $name {
            fn as_ref(&self) -> &str {
                &self.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(&self.0)
            }
        }
    };
}

id_newtype!(
    ResourceId,
    "Physical or virtual compute/memory/storage resource."
);
id_newtype!(
    TensorId,
    "Logical tensor identity (independent of physical allocation)."
);
id_newtype!(
    AllocationId,
    "Physical allocation identity (shared by views of one storage)."
);
id_newtype!(
    InstructionId,
    "Stable instruction name within an executable schedule."
);
id_newtype!(EventId, "Synchronization event identity.");
id_newtype!(RegionId, "Compiled compute-region identity.");
id_newtype!(StreamId, "Ordered stream on a resource.");
