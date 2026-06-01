#pragma once

#include <vector>

namespace correlate::dot {

void local_cpu(
  std::vector<double>& /*out NxN*/,
  const std::vector<double>& /*data MxN*/,
  const std::size_t M,
  const std::size_t N,
  const double alpha = 1.0,
  const double beta = 0.0
);

void local_corresponding_columns_cpu(
  std::vector<double>& out,
  const std::size_t offset,
  const std::vector<double>& X,
  const std::vector<double>&Y,
  const std::size_t M,
  const std::size_t N,
  const double alpha = 1.0,
  const double beta = 0.0
);

} // namespace correlate::dot
