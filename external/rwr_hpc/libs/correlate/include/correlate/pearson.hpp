#pragma once

#include <vector>

namespace correlate::pearson {

void local(
  std::vector<double>& out,
  std::vector<double>& data,
  const std::size_t M,
  const std::size_t N,
  const bool inplace = false,
  const bool use_gpu = true
);

void local_distance(
  std::vector<double>& out,
  std::vector<double>& data,
  const std::size_t M,
  const std::size_t N,
  const bool inplace = false,
  const bool use_gpu = true
);

} // namespace correlate::pearson
