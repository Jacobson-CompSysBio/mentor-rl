#include "standard_gpu.hpp"
#include <hip/hip_runtime.h>
#include <cmath>
#include <vector>
#include <stdexcept>
#include "../common/hip_common.hpp"

namespace normalize::standard {

__device__ void welford_merge(double& meanA, double& M2A, int& countA,
                              double meanB, double M2B, int countB) {
  if (countB == 0) return;

  double delta = meanB - meanA;
  int total = countA + countB;
  meanA += delta * countB / total;
  M2A += M2B + delta * delta * countA * countB / total;
  countA = total;
}

__global__ void standardize_columns_welford(double* data, std::size_t M, std::size_t N) {
  std::size_t col = blockIdx.x;
  if (col >= N) return;

  // Each thread processes a chunck of column 'col'
  std::size_t tid = threadIdx.x;
  std::size_t stride = blockDim.x;

  // Per-thread Welford stats
  double mean = 0.0;
  double M2 = 0.0;
  int count = 0;

  std::size_t offset = col * M;
  for (std::size_t i = tid; i < M; i += stride) {
    double x = data[offset + i];
    double delta = x - mean;
    ++count;
    mean += delta / count;
    M2 += delta * (x - mean);
  }

  // Reduction in shared memory
  __shared__ double s_mean[1024];
  __shared__ double s_M2[1024];
  __shared__ int    s_count[1024];

  s_mean[tid]   = mean;
  s_M2[tid]     = M2;
  s_count[tid]  = count;

  __syncthreads();

  // Reduce to thread 0
  if (tid == 0) {
    double final_mean = s_mean[0];
    double final_M2   = s_M2[0];
    int final_count   = s_count[0];

    for (int t = 1; t < blockDim.x; ++t) {
      welford_merge(final_mean, final_M2, final_count, s_mean[t], s_M2[t], s_count[t]);
    }

    double stddev = sqrt(final_M2 / (final_count - 1));
    if (stddev < 1e-8) stddev = 1.0;

    // Save to global memory for reuse
    s_mean[0] = final_mean;
    s_M2[0]   = stddev;
  }

  __syncthreads();

  // Normalize this thread's portion
  double norm_mean = s_mean[0];
  double norm_std  = s_M2[0];
  
  for (std::size_t i = tid; i < M; i += stride) {
    std::size_t idx = offset + i;
    data[idx] = (data[idx] - norm_mean) / norm_std;
  }
}

void fit_transform_device_hip(double* d_data, const std::size_t M, const std::size_t N) {
  dim3 gridDim(N);
  dim3 blockDim(128);

  standardize_columns_welford<<<gridDim, blockDim, 0, 0>>>(d_data, M, N);
  // hipDeviceSynchronize(); // Check if needed
}

void fit_transform_gpu(std::vector<double>& h_data, const std::size_t M, const std::size_t N) {
  // Verify data size matches provided dimensions
  if (h_data.size() != M * N) {
    throw std::invalid_argument("standard::fit_transform_gpu - h_data size does not equal M * N");
  }
  double* d_data;

  const std::size_t buffer_size = h_data.size() * sizeof(double);
  CHECK_HIP( hipMalloc(&d_data, buffer_size) )
  CHECK_HIP( hipMemcpy(d_data, h_data.data(), buffer_size, hipMemcpyHostToDevice) )

  fit_transform_device_hip(d_data, M, N);

  CHECK_HIP( hipMemcpy(h_data.data(), d_data, buffer_size, hipMemcpyHostToDevice) )
  CHECK_HIP( hipFree(d_data) )
}

} // namespace normalize::standard
