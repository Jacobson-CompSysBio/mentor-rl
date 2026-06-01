#include "pearson_gpu.hpp"

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include "../../common/hip_common.hpp"
#include <normalize/standard.hpp>
#include "../dot/dot_gpu.hpp"

namespace correlate::pearson {

void local_gpu(
  std::vector<double>& out,
  const std::vector<double>& data,
  const std::size_t M,
  const std::size_t N)
{
  if (data.size() != M * N) {
    throw std::invalid_argument("pearson::local_gpu - data size does not equal M * N");
  }
  if (out.size() != N * N) {
    throw std::invalid_argument("pearson::local_gpu - out size does not equal N * N");
  }

  std::size_t data_bytes = M * N * sizeof(double);
  std::size_t out_bytes  = N * N * sizeof(double);

  // Allocate device memory
  double* d_data = nullptr;
  double* d_out  = nullptr;

  CHECK_HIP(hipMalloc(&d_data, data_bytes));
  CHECK_HIP(hipMalloc(&d_out, out_bytes));
  CHECK_HIP(hipMemset(d_out, 0, out_bytes));

  // Copy data from host to device
  CHECK_HIP(hipMemcpy(d_data, data.data(), data_bytes, hipMemcpyHostToDevice));

  // Standardize data
  normalize::standard::fit_transform_device(d_data, M, N);

  // Calculate the scaled dot-product on the standardized data
  dot::local_gpu_device(d_out, d_data, M, N, 1.0 / (M-1), 0.0);

  // Copy result back to host
  CHECK_HIP(hipMemcpy(out.data(), d_out, out_bytes, hipMemcpyDeviceToHost));

  // Free memory
  CHECK_HIP(hipFree(d_data));
  CHECK_HIP(hipFree(d_out));
}

void local_distance_gpu(
  std::vector<double>& out,
  const std::vector<double>& data,
  const std::size_t M,
  const std::size_t N)
{
  if (data.size() != M * N) {
    throw std::invalid_argument("pearson::local_distance_gpu - data size does not equal M * N");
  }
  if (out.size() != N * N) {
    throw std::invalid_argument("pearson::local_distance_gpu - out size does not equal N * N");
  }

  std::size_t data_bytes = M * N * sizeof(double);
  std::size_t out_bytes  = N * N * sizeof(double);

  // Allocate device memory
  double* d_data = nullptr;
  double* d_out  = nullptr;

  CHECK_HIP(hipMalloc(&d_data, data_bytes));
  CHECK_HIP(hipMalloc(&d_out, out_bytes));
  CHECK_HIP(hipMemset(d_out, 0, out_bytes));

  // Copy data from host to device
  CHECK_HIP(hipMemcpy(d_data, data.data(), data_bytes, hipMemcpyHostToDevice));

  // Standardize data
  normalize::standard::fit_transform_device(d_data, M, N);

  // Fill d_out with all 1.0's
  fill(d_out, N * N, 1.0);

  // Calculate the scaled dot-product on the standardized data
  dot::local_gpu_device(d_out, d_data, M, N, -1.0 / (M-1), 1.0);

  // Copy result back to host
  CHECK_HIP(hipMemcpy(out.data(), d_out, out_bytes, hipMemcpyDeviceToHost));

  // Free memory
  CHECK_HIP(hipFree(d_data));
  CHECK_HIP(hipFree(d_out));
}

} // namespace correlate::pearson
