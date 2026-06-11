#include "hip_common.hpp"

namespace normalize {

bool hip_available() {
  int count = 0;
  return hipGetDeviceCount(&count) == hipSuccess && count > 0;
}

} // namespace normalize
