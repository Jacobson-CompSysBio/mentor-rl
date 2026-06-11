/// @file rank.hpp
/// @brief Utilities for ranking elements within column-major matrices.
///
/// @author Ken Smith
/// @date 2025-07-24

#pragma once

#include <vector>

/// @namespace rank
/// @brief Namespace containing functions for computing statistical ranks across vectors.
namespace rank {

enum class TieMethod {
  Min,      // lowest rank in the tie
  Max,      // hightest rank in the tie
  Average,  // mean of spanned ranks (fractional)
  Ordinal   // no ties (break arbitrarily / by second key)
};

enum class RankProgression {
  StandardCompetition,  // skip ranks (1224)
  Dense                 // no gaps (1223)
};

/// @brief Computes the ranked values of each column in a matrix with tie handling.
///
/// This function processes a column-major matrix (`data_matrix`) of size `M × N` and produces
/// a matrix of the same dimensions (`rank_matrix`) where each column's values are replaced
/// by their rank (1-based), averaged across ties.
///
/// @param[out] rank_matrix Output vector storing ranks in column-major format.
/// @param[in] data_matrix Input matrix in column-major format of size `M × N`.
/// @param[in] M Number of rows (elements per column).
/// @param[in] N Number of columns (vectors to rank).
/// @param[in] ascending If true, smaller values receive smaller ranks (default: true). If false, ranks are reversed.
///
/// @note Tied values receive the average of the ranks they span.
/// @note This function will rank each column in parallel if OpenMP is available
///
/// @throws std::invalid_argument if size of `data_matrix` does not equal `M x N`.
void rank_all_vectors(
  std::vector<double>& rank_matrix,
  const std::vector<double>& data_matrix, 
  std::size_t M,
  std::size_t N,
  TieMethod tie_method,
  RankProgression rank_method = RankProgression::StandardCompetition,
  bool ascending = true
);

} // namespace  rank
