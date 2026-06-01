#pragma once

#include <cstddef>

namespace normalize::hip::seams {

// Signature for hip_available
using HipAvailFn  = bool(*)();

extern HipAvailFn hip_available_fn;

} // namespace normalize::hip::seams
