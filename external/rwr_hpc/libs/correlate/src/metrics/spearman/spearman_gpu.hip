#include "spearman_gpu.hpp"

#include <vector>
#include <stdexcept>

#include <rank/rank.hpp>
#include "../pearson/pearson_gpu.hpp"

namespace correlate::spearman {

void local_gpu(std::vector<double>& out, const std::vector<double>& data, const std::size_t M, const std::size_t N) {
  if (data.size() != M * N) {
    throw std::invalid_argument(
      "spearman::local_gpu - data size mismatch: expected " +
      std::to_string(M * N) + ", got " + std::to_string(data.size())
    );
  }

  if (out.size() != N * N) {
    throw std::invalid_argument(
      "spearman::local__gpu - out size mismatch: expected " +
      std::to_string(N * N) + ", got " + std::to_string(out.size())
    );
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
  pearson::local_gpu(out, rank_data, M, N);
}

void local_distance_gpu(std::vector<double>& out, const std::vector<double>& data, const std::size_t M, const std::size_t N) {
  if (data.size() != M * N) {
    throw std::invalid_argument(
      "spearman::local_distance_gpu - data size mismatch: expected " +
      std::to_string(M * N) + ", got " + std::to_string(data.size())
    );
  }

  if (out.size() != N * N) {
    throw std::invalid_argument(
      "spearman::local_distance_gpu - out size mismatch: expected " +
      std::to_string(N * N) + ", got " + std::to_string(out.size())
    );
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
  pearson::local_distance_gpu(out, rank_data, M, N);
}

} // namespace correlate::spearman
