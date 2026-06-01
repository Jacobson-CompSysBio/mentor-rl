/// @file sort_utils.hpp
/// @brief Header-only library providing functor-based comparators for sorting pairs.
///
/// This library defines reusable comparator structs for sorting `std::pair` objects
/// based on either the first or second element, in increasing or decreasing order.
///
/// @author Ken Smith
/// @date 2025-07-29

#pragma once

#include <utility>
#include <numeric>  // for std::iota

/// @namespace sort
/// @brief 
namespace sort {

struct SortPairByFirstItemDecreasing {
  template<typename T, typename U>
  bool operator()(const std::pair<T, U> &lhs, const std::pair<T, U> &rhs) const {
    return lhs.first > rhs.first;
  }
};

struct SortPairByFirstItemIncreasing {
  template<typename T, typename U>
  bool operator()(const std::pair<T, U> &lhs, const std::pair<T, U> &rhs) const {
    return lhs.first < rhs.first;
  }
};

struct SortPairBySecondItemDecreasing {
  template<typename T, typename U>
  bool operator()(const std::pair<T, U> &lhs, const std::pair<T, U> &rhs) const {
    return lhs.second > rhs.second;
  }
};

struct SortPairBySecondItemIncreasing {
  template<typename T, typename U>
  bool operator()(const std::pair<T, U> &lhs, const std::pair<T, U> &rhs) const {
    return lhs.second < rhs.second;
  }
};

/// @brief Sorts two vectors in tandem based on the values in the first vector.
///
/// @tparam T Type of the `__x` vector (used for sorting).
/// @tparam U Type of the `__y` vector (reordered to follow T).
///
/// @param __x The vector whose values determine the sort order.
/// @param __y The vector to reorder in the same way as `__x`.
/// @param ascending Whether to sort in ascending or descending order (default: true).
///
/// @throws std::invalid_argument if the vectors are not the same length.
template <typename T, typename U>
void sort_two_vectors(std::vector<T>& __x,
                      std::vector<U>& __y,
                      bool ascending = true) {
  if (__x.size() != __y.size()) {
    throw std::invalid_argument("sort_two_vectors - input vectors must be of the same length");
  }

  std::vector<std::size_t> indices(__x.size());
  std::iota(indices.begin(), indices.end(), 0);

  if (ascending) {
    std::sort(indices.begin(), indices.end(),
              [&](std::size_t a, std::size_t b) { return __x[a] < __x[b]; });
  } else {
    std::sort(indices.begin(), indices.end(),
              [&](std::size_t a, std::size_t b) { return __x[a] > __x[b]; });
  }

  std::vector<T> sorted_x;
  std::vector<U> sorted_y;
  sorted_x.reserve(__x.size());
  sorted_y.reserve(__y.size());

  for (std::size_t i : indices) {
    sorted_x.push_back(__x[i]);
    sorted_y.push_back(__y[i]);
  }

  __x = std::move(sorted_x);
  __y = std::move(sorted_y);
}

/// @brief Sorts each column of a column-major matrix stored as a flattened array.
///
/// @tparam T Type of the `__x` vector (used for sorting).
///
/// @param __x Flattened matrix stored in column-major order.
/// @param num_rows Number of rows in the matrix.
/// @param num_cols Number of columns in the matrix.
/// @param ascending Whether to sort in ascending or descending order (default: true).
///
/// @throws std::invalid_argument if data.size() != num_rows * num_cols
template <typename T>
void sort_columns_in_column_major_matrix(std::vector<T>& __x,
                                         const std::size_t num_rows,
                                         const std::size_t num_cols,
                                         const bool ascending = true) {
  if (__x.size() != num_rows * num_cols) {
    throw std::invalid_argument("sort_columns_in_column_major_matrix - size mismatch");
  }

  std::vector<std::size_t> indices(num_rows);

  for (std::size_t col = 0; col < num_cols; ++col) {
    std::iota(indices.begin(), indices.end(), 0);

    if (ascending) {
      std::sort(indices.begin(), indices.end(), [&](std::size_t a, std::size_t b) {
        return __x[col * num_rows + a] < __x[col * num_rows + b];
      });
    } else {
      std::sort(indices.begin(), indices.end(), [&](std::size_t a, std::size_t b) {
        return __x[col * num_rows + a] > __x[col * num_rows + b];
      });
    }

    std::vector<double> sorted_x(num_rows);
    for (std::size_t i = 0; i < num_rows; ++i) {
      sorted_x[i] = __x[col * num_rows + indices[i]];
    }

    std::copy(sorted_x.begin(), sorted_x.end(), __x.begin() + col * num_rows);
  }
}

} // namespace sort
