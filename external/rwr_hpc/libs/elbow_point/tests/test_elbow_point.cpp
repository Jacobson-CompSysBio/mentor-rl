#include <gtest/gtest.h>
#include <gmock/gmock-matchers.h>
#include <elbow_point/elbow_point.hpp>

TEST(TestElbowPoint, ElbowPointThrowsOnX_Y_SizeMismatch) {
  std::vector<double> x = {1, 2, 3};
  std::vector<double> y= {1, 2, 3,4};
  ASSERT_THAT(
    [&](){elbow_point::elbow_point(x, y); },
    testing::ThrowsMessage<std::invalid_argument>("elbow_point - x and y must be the same size")
  );
}

TEST(TestElbowPoint, ElbowPointThrowsOnX_SizeLessThanTwo) {
  std::vector<double> x = {1};
  std::vector<double> y= {2};
  ASSERT_THAT(
    [&](){elbow_point::elbow_point(x, y); },
    testing::ThrowsMessage<std::invalid_argument>("elbow_point - x and y must have at least 2 points")
  );
}

TEST(TestElbowPoint, ElbowPointThrowsOnDuplicatesInX) {
  std::vector<double> x = {1, 1, 2, 3, 4, 5};
  std::vector<double> y= {1, 2, 3, 4, 5, 6};
  ASSERT_THAT(
    [&](){elbow_point::elbow_point(x, y); },
    testing::ThrowsMessage<std::invalid_argument>("elbow_point - x has duplicate values")
  );
}

TEST(TestElbowPoint, ElbowPointReturnsCorrectValue) {
  std::vector<double> x = {1, 2, 3, 4, 5, 6};
  std::vector<double> y= {1, 2, 3, 10, 5, 6};

  auto elbow_pt = elbow_point::elbow_point(x, y);

  EXPECT_EQ(elbow_pt.first, 4.0);
  EXPECT_EQ(elbow_pt.second, 10.0);
}