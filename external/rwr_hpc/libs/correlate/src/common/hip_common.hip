#include "hip_common.hpp"

__global__ void fill_kernel(double* data, std::size_t count, double value) {
  std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    data[idx] = value;
  }
}

namespace correlate {

bool hip_available() {
  int count = 0;
  return hipGetDeviceCount(&count) == hipSuccess && count > 0;
}

void fill(double* data, const std::size_t count, const double value) {
  int threads = 256;
  int blocks = (count + threads - 1) / threads;

  fill_kernel<<<blocks, threads>>>(data, count, value);
  CHECK_HIP(hipGetLastError());
}

}
