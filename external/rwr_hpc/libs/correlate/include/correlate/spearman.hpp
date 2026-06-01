#pragma once

#include <vector>

namespace correlate::spearman {

void local(
  std::vector<double>& out,
  const std::vector<double>& data,
  const std::size_t M,
  const std::size_t N,
  const bool use_gpu = true
);

void local_distance(
  std::vector<double>& out,
  const std::vector<double>& data,
  const std::size_t M,
  const std::size_t N,
  const bool use_gpu = true
);

} // namespace correlate::spearman
