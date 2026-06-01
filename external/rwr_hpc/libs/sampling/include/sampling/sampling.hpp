#pragma once

#include <vector>
#include <random>
#include <stdexcept>
#include <unordered_set>

namespace sampling {

/// @brief Weighted sampling without replacement using Efraimidis-Spirakis
/// @param sampled_indices [out] Output vector of sample indices (resized to n_samples)
/// @param weights Unnormalized non-negative weights
/// @param n_samples Number of samples to draw
/// @param rng A random number generator (e.g., std::mt19937_64)
/// @throws std::invalid_argument on n_samples == 0
/// @throws std::invalid_argument on less than n_samples non-zero weights
/// @throws std::invalid_argument if weights contains NaN
/// @note Items with weight == 0 are excluded from selection
void sample_weighted_without_replacement(
  std::vector<std::size_t>& sampled_indices,
  const std::vector<double>& weights,
  const std::size_t n_samples,
  std::mt19937_64& rng
);

/// @brief Draws samples with replacement from a weighted distribution.
/// @param sampled_indices [out] Output vector of sample indices (resized to n_samples)
/// @param weights Unnormalized non-negative weights
/// @param n_samples Number of samples to draw
/// @param rng A random number generator (e.g., std::mt19937_64)
/// @throws std::invalid_argument on empty weights
/// @throws std::invalid_argument on n_samples == 0
/// @throws std::invalid_argument on no non-zero weights
/// @throws std::invalid_argument if weights contains NaN
void sample_weighted_with_replacement(
  std::vector<std::size_t>& sampled_indices,
  const std::vector<double>& weights,
  const std::size_t n_samples,
  std::mt19937_64& rng
);

template <typename T>
void sample_vector_without_replacement(
  std::vector<T>& output,
  std::vector<T>& input,
  std::size_t k,
  std::mt19937_64& rng
) {
  if (k > input.size()) {
    throw std::invalid_argument("sampling::sample_vector_without_replacement - sample size cannot exceed input size.");
  }

  output.clear();
  output.reserve(k);

  std::unordered_set<std::size_t> selected_indices;
  std::uniform_int_distribution<std::size_t> dist(0, input.size() - 1);

  while (selected_indices.size() < k) {
    std::size_t idx = dist(rng);
    if (selected_indices.insert(idx).second) {
      output.push_back(input[idx]);
    }
  }
}

template <typename T>
void sample_vector_with_replacement(
  std::vector<T>& output,
  const std::vector<T>& input,
  std::size_t k,
  std::mt19937_64& rng
) {
  if (input.empty()) {
    throw std::invalid_argument("sampling::sample_vector_with_replacement - input vector must not be empty.");
  }

  std::uniform_int_distribution<std::size_t> dist(0, input.size() - 1);

  output.reserve(k);

  for (std::size_t i = 0; i < k; ++i) {
    output.push_back(input[dist(rng)]);
  }
}

} // namespace sampling
