#include "sampling/sampling.hpp"
#include <algorithm>

namespace sampling {

void sample_weighted_without_replacement(std::vector<std::size_t>& sampled_indices, const std::vector<double>& weights, const std::size_t n_samples, std::mt19937_64& rng) {
  // Verify at least one weights value provided
  if (weights.empty()) {
    throw std::invalid_argument("sampling::sample_weighted_without_replacement - weights vector is empty");
  }
  // Verify at least one sample requested
  if (n_samples == 0) {
    throw std::invalid_argument("sampling::sample_weighted_without_replacement - n_samples is 0");
  }
  // Verify at least n_samples non-zero weight
  std::size_t nnz = std::count_if(weights.begin(), weights.end(), [](double w) { return w > 0.0; });
  if (nnz < n_samples) {
    throw std::invalid_argument("sampling::sample_weighted_without_replacement - fewer non-zero weights than n_samples");
  }
  // Verify no nans
  if (std::any_of(weights.begin(), weights.end(), [](double w) { return std::isnan(w); })) {
    throw std::invalid_argument("sampling::sample_weighted_without_replacement - weights contain NaN");
  }

  std::uniform_real_distribution<double> unif(0.0, 1.0);

  std::vector<std::pair<double, std::size_t>> keys;
  keys.resize(weights.size());

  for (std::size_t i = 0; i < weights.size(); ++i) {
    double u = unif(rng);
    double key = (weights[i] > 0.0)
      ? std::pow(u, 1.0 / weights[i])
      : std::numeric_limits<double>::lowest();
    keys[i] = {key, i};
  }

  // Take top-n_samples by key
  std::nth_element(
    keys.begin(),
    keys.begin() + n_samples,
    keys.end(),
    std::greater<>()
  );

  sampled_indices.resize(n_samples);
  for (std::size_t i = 0; i < n_samples; ++i) {
    sampled_indices[i] = keys[i].second;
  }
}

void sample_weighted_with_replacement(std::vector<std::size_t>& sampled_indices, const std::vector<double>& weights, const std::size_t n_samples, std::mt19937_64& rng) {
  // Verify at least one weights value provided
  if (weights.empty()) {
    throw std::invalid_argument("sampling::sample_weighted_with_replacement - weights vector is empty");
  }
  // Verify at least one sample requested
  if (n_samples == 0) {
    throw std::invalid_argument("sampling::sample_weighted_with_replacement - n_samples is 0");
  }
  // Verify at least one non-zero weight
  if (!std::any_of(weights.begin(), weights.end(), [](double w) { return w > 0.0; })) {
    throw std::invalid_argument("sampling::sample_weighted_with_replacement - no non-zero weights to sample from");
  }
  // Verify no nans
  if (std::any_of(weights.begin(), weights.end(), [](double w) { return std::isnan(w); })) {
    throw std::invalid_argument("sampling::sample_weighted_with_replacement - weights contain NaN");
  }

  // Create a discrete probability distribution
  std::discrete_distribution<std::size_t> dist(weights.begin(), weights.end());

  // Sample from discrete probability distribution
  sampled_indices.clear();
  sampled_indices.resize(n_samples);
  for (std::size_t i = 0; i < n_samples; ++i) {
    sampled_indices[i] = dist(rng);
  }
}

} // namespace sampling
