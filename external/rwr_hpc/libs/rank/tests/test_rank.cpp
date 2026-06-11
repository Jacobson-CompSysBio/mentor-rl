#include <gtest/gtest.h>
#include <gmock/gmock-matchers.h>
#include <gmock/gmock.h>
#include <vector>
#include <cmath>  // for std::fabs

#include "../include/rank/rank.hpp"
#include "../src/rank_internal.hpp"

constexpr double EPS = 1e-6;

void expect_vectors_equal(const std::vector<double>& actual,
                          const std::vector<double>& expected,
                          double eps = EPS) {
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t i = 0; i < actual.size(); ++i) {
    EXPECT_NEAR(actual[i], expected[i], eps) << "Mismatch at index " << i;
  }
}

TEST(RankColumnWithTies, AscendingNoTies) {
  const std::size_t M = 5;
  const std::size_t col = 0;
  std::vector<double> data = {1.0, 3.0, 2.0, 5.0, 4.0}; // column 0
  std::vector<double> rank(M, 0.0);
  std::vector<std::size_t> indices(M);

  rank::rank_column_with_ties(rank, indices, data, M, col, true);

  std::vector<double> expected = {1, 3, 2, 5, 4};
  expect_vectors_equal(rank, expected);
}

TEST(RankColumnWithTies, AscendingWithTies) {
  const std::size_t M = 5;
  const std::size_t col = 0;
  std::vector<double> data = {1.0, 2.0, 2.0, 4.0, 4.0}; // column 0
  std::vector<double> rank(M, 0.0);
  std::vector<std::size_t> indices(M);

  rank::rank_column_with_ties(rank, indices, data, M, col, true);

  std::vector<double> expected = {
    1.0,         // 1.0 → rank 1
    2.5, 2.5,    // 2.0, 2.0 → average of ranks 2 & 3
    4.5, 4.5     // 4.0, 4.0 → average of ranks 4 & 5
  };
  expect_vectors_equal(rank, expected);
}

TEST(RankColumnWithTies, DescendingWithTies) {
  const std::size_t M = 5;
  const std::size_t col = 0;
  std::vector<double> data = {1.0, 2.0, 2.0, 4.0, 4.0}; // same as above
  std::vector<double> rank(M, 0.0);
  std::vector<std::size_t> indices(M);

  rank::rank_column_with_ties(rank, indices, data, M, col, false);

  std::vector<double> expected = {
    5.0,         // 1.0 → lowest
    3.5, 3.5,    // 2.0 → avg of 3 & 4
    1.5, 1.5     // 4.0 → avg of 1 & 2
  };
  expect_vectors_equal(rank, expected);
}

TEST(RankColumnWithTies, EmptyInput) {
  std::vector<double> data;
  std::vector<double> rank;
  std::vector<std::size_t> indices;

  EXPECT_NO_THROW({
    rank::rank_column_with_ties(rank, indices, data, 0, 0, true);
  });

  EXPECT_TRUE(rank.empty());
}

TEST(RankColumnWithTies, ConstantColumn) {
  const std::size_t M = 4;
  std::vector<double> data = {1.0, 1.0, 1.0, 1.0};
  std::vector<double> rank(M, 0.0);
  std::vector<std::size_t> indices(M);

  rank::rank_column_with_ties(rank, indices, data, M, 0, true);

  std::vector<double> expected(M, 2.5);  // avg of ranks 1+2+3+4 = 10 / 4 = 2.5
  expect_vectors_equal(rank, expected);
}

TEST(RankAllVectors, ThrowsOnSizeMismatch) {
  std::vector<double> rank_data;
  std::vector<double> in_data(50);
  const std::size_t M = 10, N = 6;

  ASSERT_THAT(
    [&](){rank::rank_all_vectors(rank_data, in_data, M, N); },
    testing::ThrowsMessage<std::invalid_argument>("rank_all_vectors - data_matrix size does not equal M * N")
  );
}

TEST(RankAllVectors, MultipleColumns) {
  const std::size_t M = 3;
  const std::size_t N = 2;

  // Column-major: col 0 = {3, 1, 2}, col 1 = {5, 4, 6}
  std::vector<double> data = {
    3, 1, 2,  // col 0
    5, 4, 6   // col 1
  };

  std::vector<double> rank;

  rank::rank_all_vectors(rank, data, M, N, true);

  std::vector<double> expected = {
    3.0, 1.0, 2.0,  // col 0
    2.0, 1.0, 3.0   // col 1
  };
  expect_vectors_equal(rank, expected);
}

TEST(RankAllVectors, SizeMismatch) {
  const std::size_t M = 3;
  std::vector<double> data = {1.0, 2.0};  // Too short
  std::vector<double> rank(M, 0.0);
  std::vector<std::size_t> indices(M);

  ASSERT_THAT(
    [&](){rank::rank_all_vectors(rank, data, M, 0, true); },
    testing::ThrowsMessage<std::invalid_argument>("rank_all_vectors - data_matrix size does not equal M * N")
  );
}

TEST(RankAllVectors, ConstantColumns) {
  const std::size_t M = 3;
  const std::size_t N = 2;

  std::vector<double> data = {
    7, 7, 7,  // col 0
    2, 2, 2   // col 1
  };

  std::vector<double> rank;

  rank::rank_all_vectors(rank, data, M, N, true);

  std::vector<double> expected(M * N, 2.0);  // average of ranks 1+2+3 = 6 / 3 = 2
  expect_vectors_equal(rank, expected);
}
