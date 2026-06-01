/// @file vector_utils.hpp
/// @brief Header-only utility functions for common operations on std::vector.
///
/// Provides template-based utilities for detecting duplicates, finding extrema indices,
/// and removing elements from vectors. All functions are implemented inline in the `utils` namespace.
///
/// @author Ken Smith
/// @date 2025-07-25

#pragma once

#include <vector>
#include <unordered_set>
#include <algorithm>
#include <iterator>
#include <stdexcept>
#include <limits>
#include <type_traits>
#include <string>

/// @namespace utils
/// @brief Utility functions for vector and container operations.
namespace utils {

/// @brief Checks if a vector contains any duplicate elements.
///
/// @tparam T Type of elements in the vector (must be hashable and comparable).
///
/// @param __x The input vector to check.
///
/// @return true if any duplicate exists, false otherwise.
template<typename T>
bool has_duplicates(const std::vector<T>& __x) {
  std::unordered_set<T> unique_elements;

  for (T element : __x) {
    if (unique_elements.count(element)) { // If element already in the set
      return true;
    }
    unique_elements.insert(element); // Add element to the set
  }
  return false; // No duplicates found
}

/// @brief Returns the index of the minimum element in a vector.
///
/// @tparam T Type of elements in the vector (must support operator<).
///
/// @param __x The input vector.
///
/// @return Index of the smallest element in the vector.
template<typename T>
std::size_t idx_of_min_element(const std::vector<T>& __x){
  auto min_it = std::min_element(__x.begin(), __x.end());
  return std::distance(__x.begin(), min_it);
}

/// @brief Returns the index of the maximum element in a vector.
///
/// @tparam T Type of elements in the vector (must support operator>).
///
/// @param __x The input vector.
///
/// @return Index of the largest element in the vector.
template<typename T>
std::size_t idx_of_max_element(const std::vector<T>& __x){
  auto min_it = std::max_element(__x.begin(), __x.end());
  return std::distance(__x.begin(), min_it);
}

/// @brief Removes all elements from vector `__x` that also appear in vector `__y`.
///
/// This function performs in-place filtering of `__x`, removing any values that exist in `__y`.
///
/// @tparam T Type of elements in the vectors (must be hashable and comparable).
///
/// @param __x The vector to modify (in-place).
/// @param __y The vector containing elements to remove from `__x`.
template<typename T>
void remove_elements(std::vector<T>& __x, const std::vector<T> &__y) {
  std::unordered_set<T> remove_set(__y.begin(), __y.end());
  __x.erase(
    std::remove_if(
      __x.begin(),
      __x.end(), 
      [&](T val) {return remove_set.count(val) > 0; }
    ),
    __x.end());
}

/// @brief Safely cast a std::vector<T> to std::vector<U>
///
/// @tparam T Input type (e.g., size_t)
/// @tparam U Output type (e.g., int)
///
/// @param input The input vector to convert
/// 
/// @return A new std::vector<U> containing casted values from input
///
/// @throws std::overflow_error if any value in input cannot be safely represented as U
template <typename U, typename T>
std::vector<U> safe_cast_vector(const std::vector<T>& input) {
  static_assert(std::is_arithmetic<T>::value, "T must be numeric");
  static_assert(std::is_arithmetic<U>::value, "U must be numeric");

  std::vector<U> output(input.size());

  for (size_t i = 0; i < input.size(); ++i) {
    const T& val = input[i];

    // Check for NaN or Inf if T is floating-point
    if constexpr (std::is_floating_point<T>::value) {
      if (std::isnan(val)) {
        throw std::runtime_error("Element " + std::to_string(i) + ": Cannot convert NaN to integral type");
      }
      if (std::isinf(val)) {
        throw std::runtime_error("Element " + std::to_string(i) + ": Cannot convert infinity to integral type");
      }
    }

    // Check for negative signed → unsigned conversion
    if constexpr (std::is_signed<T>::value && std::is_unsigned<U>::value) {
      if (val < 0) {
        throw std::runtime_error("Element " + std::to_string(i) + ": Negative value " +
                                 std::to_string(val) + " cannot be cast to unsigned type");
      }
    }

    // General range check using long double to avoid truncation/wraparound
    long double val_ld = static_cast<long double>(val);
    long double tgt_min = static_cast<long double>(std::numeric_limits<U>::lowest());
    long double tgt_max = static_cast<long double>(std::numeric_limits<U>::max());

    if (val_ld < tgt_min || val_ld > tgt_max) {
      std::ostringstream oss;
      oss << "Element " << i << ": Value " << val << " out of range for target type ["
          << tgt_min << ", " << tgt_max << "]";
      throw std::runtime_error(oss.str());
    }

    // All checks passed — cast safely
    output[i] = static_cast<U>(val);
  }

  return output;
}

std::string concate(const std::vector<std::string>& __x, const std::string& sep = "_");

} // namespace utils
