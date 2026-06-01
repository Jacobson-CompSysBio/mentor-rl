#include "spearman_cpu.hpp"
#include <stdexcept>

#include <rank/rank.hpp>
#include "../pearson/pearson_cpu.hpp"

namespace correlate::spearman {

void local_cpu(
  std::vector<double>& out,
  const std::vector<double>& data,
  const std::size_t M,
  const std::size_t N
) {
  if (data.size() != M * N) {
    throw std::invalid_argument("spearman::local_cpu - data size does not equal M * N");
  }
  if (out.size() != N * N) {
    throw std::invalid_argument("spearman::local_cpu - out size does not equal N * N");
  }

  // Calculate the rank data
  std::vector<double> rank_data;
  rank::rank_all_vectors(
    rank_data,
    data,
    M,
    N,
    rank::TieMethod::Average
  );

  // Pearson's on the ranked data is spearman's
  pearson::local_cpu(out, rank_data, M, N, true);
}

void local_distance_cpu(
  std::vector<double>& out,
  const std::vector<double>& data,
  const std::size_t M,
  const std::size_t N
) {
  if (data.size() != M * N) {
    throw std::invalid_argument("spearman::local_distance_cpu - data size does not equal M * N");
  }
  if (out.size() != N * N) {
    throw std::invalid_argument("spearman::local_distance_cpu - out size does not equal N * N");
  }

  // Calculate the rank data
  std::vector<double> rank_data;
  rank::rank_all_vectors(
    rank_data,
    data,
    M,
    N,
    rank::TieMethod::Average
  );

  // Pearson's on the ranked data is spearman's
  // Pearson's distance intitialized out to all 1.0's
  pearson::local_distance_cpu(out, rank_data, M, N, true);
}

} // namespace correlate::spearman
