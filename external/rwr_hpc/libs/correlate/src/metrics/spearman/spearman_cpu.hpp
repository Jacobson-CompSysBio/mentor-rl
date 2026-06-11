#pragma once

#include <vector>

namespace correlate::spearman {

void local_cpu(
  std::vector<double>& out /*NxN*/,
  const std::vector<double>& data /*MxN*/,
  const std::size_t M,
  const std::size_t N
);

void local_distance_cpu(
  std::vector<double>& out /*NxN*/,
  const std::vector<double>& data /*MxN*/,
  const std::size_t M,
  const std::size_t N
);

} // namespace correlate::spearman
