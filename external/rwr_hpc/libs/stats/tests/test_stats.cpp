#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <stats/stats.hpp>

TEST(TestStats, ComputeMedianThrowsOnEmptyVector) {
  std::vector<double> data;
  ASSERT_THAT(
    [&](){stats::compute_median(data); },
    testing::ThrowsMessage<std::invalid_argument>("compute_median: input vector is empty")
  );
}

TEST(TestStats, ComputeMedianWithInputofSizeOne) {
  std::vector<double> data = {5.0};
  const double expected_median = 5.0;
  double actual_median = stats::compute_median(data, false);

  EXPECT_EQ(actual_median, expected_median);
}

TEST(TestStats, ComputeMedianCopyEvenNumberOfElements) {
  std::vector<double> data = {1, 36, 24, 2, 0, -1};
  const double expected_median = 1.5;
  const auto expected_data = data;

  double actual_median = stats::compute_median(data, false);

  EXPECT_NEAR(expected_median, actual_median, 1e-8);
  EXPECT_EQ(data, expected_data);
}

TEST(TestStats, ComputeMedianCopyOddNumberOfElements) {
  std::vector<double> data = {0.4, 6, 13, 8484, 243, 2, 8};
  const double expected_median = 8.0;
  const auto expected_data = data;

  double actual_median = stats::compute_median(data, false);

  EXPECT_NEAR(expected_median, actual_median, 1e-8);
  EXPECT_EQ(data, expected_data);
}

TEST(TestStats, ComputeMedianInplaceEvenNumberOfElements) {
  std::vector<double> data = {4, 7, 13, 93, 12, 4};
  std::vector<double> original = data;
  const double expected_median = 9.5;

  double actual_median = stats::compute_median(data, true);

  EXPECT_NEAR(expected_median, actual_median, 1e-8);

  // Check content preserved (unordered)
  EXPECT_THAT(data, ::testing::UnorderedElementsAreArray(original));
}

TEST(TestStats, ComputeMedianInplaceOddNumberOfElements) {
  std::vector<double> data = {499, 40, 234, 43, 1, 3, 4};
  std::vector<double> original = data;
  const double expected_median = 40.0;

  double actual_median = stats::compute_median(data, true);

  EXPECT_NEAR(expected_median, actual_median, 1e-8);

  // Check content preserved (unordered)
  EXPECT_THAT(data, ::testing::UnorderedElementsAreArray(original));
}