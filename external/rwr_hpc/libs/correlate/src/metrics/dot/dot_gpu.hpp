#pragma once

#include <vector>

namespace correlate::dot {

void local_gpu(
  std::vector<double>& out /*out NxN*/,
  const std::vector<double>& data /*data MxN*/,
  const std::size_t M,
  const std::size_t N,
  const double alpha = 1.0,
  const double beta = 0.0,
  const double initial_output = 0.0
);

void local_gpu_device(
  double* d_out,
  const double* d_in,
  const std::size_t M,
  const std::size_t N,
  const double alpha = 1.0,
  const double beta = 0.0
);

} // namespace correlate::dot
