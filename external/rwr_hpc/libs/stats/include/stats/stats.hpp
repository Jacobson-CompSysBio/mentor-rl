/// @file median.hpp
/// @brief Header-only utility for computing the median of a numeric vector.
///
/// Provides a generic median computation function using `std::nth_element`
/// to achieve linear average-case performance.

#pragma once

#include <vector>
#include <algorithm>
#include <type_traits>
#include <stdexcept>

/// @namespace stats
/// @brief Statistical utility functions for numeric containers.
namespace  stats {

/// @brief Computes the median value of a vector of numeric elements.
///
/// If `in_place` is false (default), the function operates on a copy of the input vector
/// so that the original data remains unchanged.
///
/// If `in_place` is true, the function will modify the input vector in-place
/// to avoid unnecessary copying.
///
/// @tparam T Numeric type (must satisfy std::is_arithmetic).
///
/// @param __x A vector of values (copied unless `in_place = true`).
/// @param in_place Flag to determine whether to compute in-place or on a copy (defualt: false).
///
/// @return The median value of the vector.
///
/// @throws std::runtime_error if the input vector is empty.
///
/// @note Uses `std::nth_element` for average linear-time performance.
///
/// @example
/// std::vector<double> values = {4.0, 1.5, 3.0};
/// double med = stats::compute_median(values);  // returns 3.0
template <typename T>
T compute_median(std::vector<T> __x, const bool in_place = false) {
  static_assert(std::is_arithmetic<T>::value, "compute_median requires a numeric type");

  if (__x.empty()) {
    throw std::invalid_argument("compute_median: input vector is empty");
  }

  if (__x.size() == 1) {
    return __x[0];
  }
  
  std::vector<T>& x = in_place ? __x : ( __x = std::vector<T>(__x));  // copy if not in-place

  std::size_t n = x.size();
  std::size_t mid = n / 2;

  std::nth_element(x.begin(), x.begin() + mid, x.end());

  if (n % 2 == 1) {
    return x[mid];
  } else {
    T mid1 = x[mid];
    std::nth_element(x.begin(), x.begin() + mid - 1, x.end());
    T mid2 = x[mid - 1];
    return (mid1 + mid2) / static_cast<T>(2);
  }
}

} // namespace  stats
