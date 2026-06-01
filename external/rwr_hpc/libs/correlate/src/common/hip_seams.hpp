#pragma once

#include <cstddef>

namespace correlate::hip::seams {

// Signature for hip_available
using HipAvailFn  = bool(*)();

extern HipAvailFn hip_available_fn;

} // namespace correlate::hip::seams
