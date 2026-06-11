#pragma once

#include <vector>

namespace correlate::pearson {

void local_cpu(
  std::vector<double>& /*out NxN*/,
  std::vector<double>& /*data MxN*/,
  const std::size_t M,
  const std::size_t N,
  const bool inplace = true
);

void local_distance_cpu(
  std::vector<double>& /*out NxN*/,
  std::vector<double>& /*data MxN*/,
  const std::size_t M,
  const std::size_t N,
  const bool inplace = true
);

} // namespace correlate::pearson
