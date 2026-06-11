#include "pearson_cpu.hpp"
#include <stdexcept>

#include <normalize/standard.hpp>
#include "../dot/dot_cpu.hpp"

namespace correlate::pearson {

void local_cpu(
  std::vector<double>& out,
  std::vector<double>& data,
  const std::size_t M,
  const std::size_t N,
  const bool inplace
) {
  if (data.size() != M * N) {
    throw std::invalid_argument("pearson::local_cpu - data size does not equal M * N");
  }
  if (out.size() != N * N) {
    throw std::invalid_argument("pearson::local_cpu - out size does not equal N * N");
  }

  if (inplace) {
    normalize::standard::fit_transform(data, M, N, false);
    dot::local_cpu(out, data, M, N, 1.0 / (M - 1), 0.0);
  } else {
    // Create local copy of data
    auto local_data = data;
    normalize::standard::fit_transform(local_data, M, N, false);
    dot::local_cpu(out, local_data, M, N, 1.0 / (M - 1), 0.0);
  }
}

void local_distance_cpu(
  std::vector<double>& out,
  std::vector<double>& data,
  const std::size_t M,
  const std::size_t N,
  const bool inplace
) {
  if (data.size() != M * N) {
    throw std::invalid_argument("pearson::local_distance_cpu - data size does not equal M * N");
  }
  if (out.size() != N * N) {
    throw std::invalid_argument("pearson::local_distance_cpu - out size does not equal N * N");
  }

  for (auto& o : out) { o = 1.0; }

  if (inplace) {
    normalize::standard::fit_transform(data, M, N, false);
    dot::local_cpu(out, data, M, N, -1.0 / (M - 1), 1.0);
  } else {
    // Create local copy of data
    auto local_data = data;
    normalize::standard::fit_transform(local_data, M, N, false);
    dot::local_cpu(out, local_data, M, N, -1.0 / (M - 1), 1.0);
  }
}

} // namespace correlate::pearson
