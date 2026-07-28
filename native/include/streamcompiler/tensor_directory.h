#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace streamcompiler {

enum class MemoryTier : uint8_t {
  Vram = 0,
  PinnedHost = 1,
  NumaRam = 2,
  UnifiedShared = 3,
  Nvme = 4,
  DiskCache = 5,
};

struct ResidentCopy {
  std::string memory;
  uint64_t version = 0;
  uint64_t nbytes = 0;
};

// Logical tensor identity with versioned resident copies.
// Stale copies must be invalidated after mutations.
struct TensorRecord {
  std::string id;
  MemoryTier home = MemoryTier::NumaRam;
  std::vector<ResidentCopy> copies;
  uint64_t version = 0;
  bool immutable = true;
};

}  // namespace streamcompiler
