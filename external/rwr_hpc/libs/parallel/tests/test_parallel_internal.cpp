#include "gtest/gtest.h"
#include <gmock/gmock-matchers.h>
#include <stdexcept>
#include "parallel_internal.hpp"

TEST(ParallelInternalTest, CalculateDisplacmentCorrectly) {
  const std::vector<int> counts = {1,2,3,4,5};

  auto displ = parallel::calc_displacement(counts);

  EXPECT_EQ(displ.size(), counts.size());
  EXPECT_EQ(displ[0], 0);
  EXPECT_EQ(displ[1], 1);
  EXPECT_EQ(displ[2], 3);
  EXPECT_EQ(displ[3], 6);
  EXPECT_EQ(displ[4], 10);
}