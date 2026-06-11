#pragma once

#include <vector>

namespace correlate::pearson {

void local_gpu(
  std::vector<double>& out /*out NxN*/,
  const std::vector<double>& data /*data MxN*/,
  const std::size_t M,
  const std::size_t N
);

void local_distance_gpu(
  std::vector<double>& out /*out NxN*/,
  const std::vector<double>& data /*data MxN*/,
  const std::size_t M,
  const std::size_t N
);

} // namespace correlate::pearson
