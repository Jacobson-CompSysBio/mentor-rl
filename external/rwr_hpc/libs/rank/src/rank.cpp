#include <rank/rank.hpp>
#include "rank_internal.hpp"
#include <algorithm>
#include <numeric>
#include <stdexcept>

#ifdef USE_OPENMP
#include <omp.h>
#endif

namespace rank {
  
void rank_column_with_ties(
  std::vector<double>& rank_matrix,
  std::vector<std::size_t>& indices,
  const std::vector<double>& data_matrix,
  std::size_t M,
  std::size_t col,
  TieMethod tie_method,
  RankProgression rank_method,
  bool ascending)
{
  // Fill indices vector starting at 0
  for (std::size_t i = 0; i < M; ++i) {
    indices[i] = i;
  }

  // Save local calculation
  const std::size_t col_M = col * M;

  // Sort indices based on values in the column
  std::stable_sort(
    indices.begin(),
    indices.end(),
    [&](std::size_t a, std::size_t b) {
      if (ascending) {
        return data_matrix[col_M + a] < data_matrix[col_M + b];
      } else {
        return data_matrix[col_M + a] > data_matrix[col_M + b];
      }
    }
  );

  // Ordinal: no ties, rank strictly increases
  if (tie_method == TieMethod::Ordinal) {
    for (std::size_t k = 0; k < M; ++k) {
      rank_matrix[col_M + indices[k]] = static_cast<double>(k + 1);
    }
    return;
  }

  std::size_t current_rank = 1;
  std::size_t i = 0;

  while (i < M) {
    std::size_t j = i + 1;

    // Find tie range
    while (j < M && data_matrix[col_M + indices[i]] == data_matrix[col_M + indices[j]]) {
      ++j;
    }

    std::size_t tie_count = j - i;

    double assigned_rank = 0.0;

    switch (tie_method) {
    case TieMethod::Min:
      assigned_rank = static_cast<double>(current_rank);
      break;
    
    case TieMethod::Max:
      assigned_rank = static_cast<double>(current_rank + tie_count - 1);
      break;
    
    case TieMethod::Average:
      assigned_rank = static_cast<double>(current_rank) +
                      (static_cast<double>(tie_count) - 1.0) / 2.0;
      break;
    }

    // Assign rank to all tied rows
    for (std::size_t k = i; k < j; ++k) {
      rank_matrix[col_M + indices[k]] = assigned_rank;
    }

    // Advance rank counter
    if (rank_method == RankProgression::StandardCompetition) {
      current_rank += tie_count;
    } else { // Dense
      current_rank += 1;
    }

    i = j; // Move i to the next set of unique values
  }
}

void rank_all_vectors(
  std::vector<double>& rank_matrix,
  const std::vector<double>& data_matrix,
  std::size_t M,
  std::size_t N,
  TieMethod tie_method,
  RankProgression rank_method,
  bool ascending)
{
  const std::size_t MN = M * N;

  // Verify input dimensions
  if (data_matrix.size() != MN) {
    throw std::invalid_argument(
      "rank_all_vectors - data_matrix size does not equal M * N"
    );
  }
  // Validate tie_method and rank_method combination
  if (tie_method == TieMethod::Average && rank_method == RankProgression::Dense) {
    throw std::logic_error(
      "rank_all_vectors - Average tie method with dense ranking is undefined"
    );
  }

  if(M == 0 || N == 0) {
    rank_matrix.clear();
    return;
  }

  // Ensure output is sized correctly
  rank_matrix.resize(MN);

  #ifdef USE_OPENMP
  #pragma omp parallel for schedule(static)
  #endif
  {
    for (std::size_t col = 0; col < N; ++col) {
    // Allocate thread-private scratch buffer once per thread
    std::vector<std::size_t> indices(M);

    rank_column_with_ties(rank_matrix,
                          indices,
                          data_matrix,
                          M,
                          col,
                          tie_method,
                          rank_method,
                          ascending);
    }
  }
}

} // namespace rank
