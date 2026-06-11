#include "dot_gpu.hpp"

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include "../../common/hip_common.hpp"

namespace correlate::dot {

void local_gpu(
  std::vector<double>& out /*out NxN*/,
  const std::vector<double>& data /*data MxN*/,
  const std::size_t M,
  const std::size_t N,
  const double alpha,
  const double beta,
  const double initial_output
) {
  if (data.size() != M * N) {
    throw std::invalid_argument("dot::local_gpu - data size does not equal M * N");
  }
  if (out.size() != N * N) {
    throw std::invalid_argument("dot::local_gpu - out size does not equal N * N");
  }

  std::size_t data_bytes = data.size() * sizeof(double);
  std::size_t out_bytes  = out.size() * sizeof(double);

  // Allocate device memory
  double* d_data = nullptr;
  double* d_out  = nullptr;

  CHECK_HIP(hipMalloc(&d_data, data_bytes));
  CHECK_HIP(hipMalloc(&d_out, out_bytes));

  // Fill d_out
  correlate::fill(d_out, out.size(), initial_output);

  // Copy data from host to device
  CHECK_HIP(hipMemcpy(d_data, data.data(), data_bytes, hipMemcpyHostToDevice));

  local_gpu_device(d_out, d_data, M, N, alpha, beta);

  // Copy result back to host
  CHECK_HIP(hipMemcpy(out.data(), d_out, out_bytes, hipMemcpyDeviceToHost));

  // Free memory
  CHECK_HIP(hipFree(d_data));
  CHECK_HIP(hipFree(d_out));
}

void local_gpu_device(
  double* d_out,
  const double* d_in,
  const std::size_t M,
  const std::size_t N,
  const double alpha,
  const double beta
) {
  // Initialize hipBLAS
  hipblasHandle_t handle;
  CHECK_HIPBLAS(hipblasCreate(&handle));

  CHECK_HIPBLAS( hipblasDgemm(
    handle,
    HIPBLAS_OP_T,
    HIPBLAS_OP_N,
    N, N, M,
    &alpha,
    d_in, M,
    d_in, M,
    &beta,
    d_out, N
  ));

  CHECK_HIPBLAS(hipblasDestroy(handle));
}

} // namespace correlate::dot

