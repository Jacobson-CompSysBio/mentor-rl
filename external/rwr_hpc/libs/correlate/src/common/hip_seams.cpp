#include "hip_seams.hpp"

#include "hip_common.hpp"

namespace correlate::hip::seams {

// Initialize the seams to point to the real implementation
HipAvailFn hip_available_fn = &hip_available;

} // namespace correlate::hip::seams
