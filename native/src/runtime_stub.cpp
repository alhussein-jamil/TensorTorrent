// Placeholder native runtime entry. Real stream/event/IO executors land here
// without embedding CUDA-only assumptions in the public headers.
#include "streamcompiler/runtime.h"

namespace streamcompiler {

const char* runtime_version() { return "0.1.0-stub"; }

}  // namespace streamcompiler
