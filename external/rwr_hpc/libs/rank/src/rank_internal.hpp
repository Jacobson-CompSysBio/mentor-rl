/// @internal
/// @file rank_internal.hpp
/// @brief Internal utilities for ranking elements within column-major matrices.
///
/// @author Ken Smith
/// @date 2025-07-24 


#pragma once

#include <vector>

namespace rank {

void rank_column_with_ties(
  std::vector<double>& rank_matrix,
  std::vector<std::size_t>& indices,
  const std::vector<double>& data_matrix,
  std::size_t N,
  std::size_t col,
  bool ascending = true
);

} // namespace rank
