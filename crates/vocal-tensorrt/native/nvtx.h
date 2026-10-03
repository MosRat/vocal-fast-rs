#pragma once

// NVTX 3 is header-only and does not add a link-time dependency.  The
// annotations are consumed by Nsight Systems/Compute and are harmless when no
// profiler is attached.
#include <nvtx3/nvtx3.hpp>
