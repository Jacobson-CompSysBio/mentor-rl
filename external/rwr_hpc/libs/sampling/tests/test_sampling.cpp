#include <gtest/gtest.h>
#include <gmock/gmock-matchers.h>
#include <random>
#include <unordered_set>
#include "sampling/sampling.hpp"

TEST(SampleWeightedWithoutReplacementTest, ThrowsOnEmptyWeights) {
  std::vector<double> weights;
  std::vector<std::size_t> indices;
  std::mt19937_64 rng(42);

  ASSERT_THAT(
    [&](){sampling::sample_weighted_without_replacement(indices, weights, 10, rng); },
    testing::ThrowsMessage<std::invalid_argument>("sampling::sample_weighted_without_replacement - weights vector is empty")
  );
}

TEST(SampleWeightedWithoutReplacementTest, ThrowsOnN_SampleZero) {
  std::vector<double> weights = {0.5, 0.5};
  std::vector<std::size_t> indices;
  std::mt19937_64 rng(42);

  ASSERT_THAT(
    [&](){sampling::sample_weighted_without_replacement(indices, weights, 0, rng); },
    testing::ThrowsMessage<std::invalid_argument>("sampling::sample_weighted_without_replacement - n_samples is 0")
  );
}

TEST(SampleWeightedWithoutReplacementTest, ThrowsOnTooFewNnzWeights) {
  std::vector<double> weights = {0.5, 0.5, 0.0, 0.0, 0.0};
  std::vector<std::size_t> indices;
  std::mt19937_64 rng(42);

  ASSERT_THAT(
    [&](){sampling::sample_weighted_without_replacement(indices, weights, 3, rng); },
    testing::ThrowsMessage<std::invalid_argument>("sampling::sample_weighted_without_replacement - fewer non-zero weights than n_samples")
  );
}

TEST(SampleWeightedWithoutReplacementTest, ThrowsOnWeightContainsNan) {
  std::vector<double> weights = {0.5, 0.3, std::nan(""), 0.3};
  std::vector<std::size_t> indices;
  std::mt19937_64 rng(42);

  ASSERT_THAT(
    [&](){sampling::sample_weighted_without_replacement(indices, weights, 3, rng); },
    testing::ThrowsMessage<std::invalid_argument>("sampling::sample_weighted_without_replacement - weights contain NaN")
  );
}

TEST(SampleWeightedWithoutReplacementTest, GeneratesCorrectSizeAndBounds) {
  std::vector<double> weights = {0.1, 0.2, 0.3, 0.05, 0.05, 0.05, 0.05, 0.2};
  std::vector<size_t> indices;
  std::mt19937_64 rng(42);

  sampling::sample_weighted_without_replacement(indices, weights, 5, rng);

  ASSERT_EQ(indices.size(), 5);
  for (auto i : indices) {
    EXPECT_LT(i, weights.size());
  }
}

TEST(SampleWeightedWithoutReplacementTest, ReproducibleResults) {
  std::vector<double> weights = {0.1, 0.1, 0.1, 0.3, 0.4};
  std::vector<size_t> indices1, indices2;
  std::mt19937_64 rng1(1234), rng2(1234);

  sampling::sample_weighted_without_replacement(indices1, weights, 3, rng1);
  sampling::sample_weighted_without_replacement(indices2, weights, 3, rng2);

  EXPECT_EQ(indices1, indices2);
}

TEST(SampleWeightedWithoutReplacementTest, UniqueIndicesReturned) {
  std::vector<double> weights = {0.5, 0.5, 0.5, 0.5, 0.5};
  std::vector<size_t> indices;
  std::mt19937_64 rng(42);

  sampling::sample_weighted_without_replacement(indices, weights, 5, rng);

  std::unordered_set<size_t> unique(indices.begin(), indices.end());
  EXPECT_EQ(unique.size(), indices.size());  // no duplicates
}


TEST(SampleWeightedWithReplacementTest, ThrowsOnEmptyWeights) {
  std::vector<double> weights;
  std::vector<std::size_t> indices;
  std::mt19937_64 rng(42);

  ASSERT_THAT(
    [&](){sampling::sample_weighted_with_replacement(indices, weights, 10, rng); },
    testing::ThrowsMessage<std::invalid_argument>("sampling::sample_weighted_with_replacement - weights vector is empty")
  );
}

TEST(SampleWeightedWithReplacementTest, ThrowsOnN_SampleZero) {
  std::vector<double> weights = {0.5, 0.5};
  std::vector<std::size_t> indices;
  std::mt19937_64 rng(42);

  ASSERT_THAT(
    [&](){sampling::sample_weighted_with_replacement(indices, weights, 0, rng); },
    testing::ThrowsMessage<std::invalid_argument>("sampling::sample_weighted_with_replacement - n_samples is 0")
  );
}

TEST(SampleWeightedWithReplacementTest, ThrowsOnTooFewNnzWeights) {
  std::vector<double> weights = {0.0, 0.0, 0.0, 0.0, 0.0};
  std::vector<std::size_t> indices;
  std::mt19937_64 rng(42);

  ASSERT_THAT(
    [&](){sampling::sample_weighted_with_replacement(indices, weights, 3, rng); },
    testing::ThrowsMessage<std::invalid_argument>("sampling::sample_weighted_with_replacement - no non-zero weights to sample from")
  );
}

TEST(SampleWeightedWithReplacementTest, ThrowsOnWeightContainsNan) {
  std::vector<double> weights = {0.5, 0.3, std::nan(""), 0.3};
  std::vector<std::size_t> indices;
  std::mt19937_64 rng(42);

  ASSERT_THAT(
    [&](){sampling::sample_weighted_with_replacement(indices, weights, 3, rng); },
    testing::ThrowsMessage<std::invalid_argument>("sampling::sample_weighted_with_replacement - weights contain NaN")
  );
}

TEST(SampleWeightedWithReplacementTest, GeneratesCorrectSizeAndBounds) {
  std::vector<double> weights = {0.1, 0.2, 0.3, 0.05, 0.05, 0.05, 0.05, 0.2};
  std::vector<size_t> indices;
  std::mt19937_64 rng(42);

  sampling::sample_weighted_with_replacement(indices, weights, 5, rng);

  ASSERT_EQ(indices.size(), 5);
  for (auto i : indices) {
    EXPECT_LT(i, weights.size());
  }
}

TEST(SampleWeightedWithReplacementTest, ReproducibleResults) {
  std::vector<double> weights = {0.1, 0.1, 0.1, 0.3, 0.4};
  std::vector<size_t> indices1, indices2;
  std::mt19937_64 rng1(1234), rng2(1234);

  sampling::sample_weighted_with_replacement(indices1, weights, 3, rng1);
  sampling::sample_weighted_with_replacement(indices2, weights, 3, rng2);

  EXPECT_EQ(indices1, indices2);
}

TEST(SampleVectorWithoutReplacement, ThrowsWhenK_GreaterThanInputSize) {
  std::vector<std::string> output;
  std::vector<std::string> input = {"node1", "node2", "node3", "node4"};
  std::mt19937_64 rng(42);

  ASSERT_THAT(
    [&](){sampling::sample_vector_without_replacement(output, input, 10, rng); },
    testing::ThrowsMessage<std::invalid_argument>("sampling::sample_vector_without_replacement - sample size cannot exceed input size.")
  );
}

TEST(SampleVectorWithoutReplacement, CreatesK_UniqueSamples) {
  std::vector<std::string> output;
  std::vector<std::string> input = {"node1", "node2", "node3", "node4"};
  std::mt19937_64 rng(42);

  sampling::sample_vector_without_replacement(output, input, 3, rng);

  std::unordered_set<std::string> allowed_set = {"node1", "node2", "node3", "node4"};

  ASSERT_EQ(output.size(), 3);
  for (std::size_t i = 0; i < 3; ++i) {
    std::unordered_set<std::string> seen;

    for (const auto& o : output) {
      EXPECT_TRUE(allowed_set.count(o)); // found in node labels

      bool inserted = seen.insert(o).second;
      EXPECT_TRUE(inserted); // Unique in current vector
    }
  }
}

TEST(SampleVectorWithReplacement, ThrowsOfInputIsEmpty) {
  std::vector<std::string> output;
  std::vector<std::string> input;
  std::mt19937_64 rng(42);

  ASSERT_THAT(
    [&](){sampling::sample_vector_with_replacement(output, input, 10, rng); },
    testing::ThrowsMessage<std::invalid_argument>("sampling::sample_vector_with_replacement - input vector must not be empty.")
  );
}

TEST(SampleVectorWithReplacement, CreatesK_ValidSamples) {
  std::vector<std::string> output;
  std::vector<std::string> input = {"node1", "node2", "node3", "node4"};
  std::mt19937_64 rng(42);

  sampling::sample_vector_with_replacement(output, input, 5, rng);

  std::unordered_set<std::string> allowed_set = {"node1", "node2", "node3", "node4"};

  ASSERT_EQ(output.size(), 5);
  for (std::size_t i = 0; i < 5; ++i) {
    for (const auto& o : output) {
      EXPECT_TRUE(allowed_set.count(o)); // found in node labels
    }
  }
}
