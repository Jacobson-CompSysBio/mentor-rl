#include <gtest/gtest.h>
#include <gmock/gmock-matchers.h>
#include <sort/sort.hpp>

TEST(TestSort, SortColumnsInColumnMajorMatrixThrowsOnSizeMismatch) {
  std::vector<double> data = {1.0, 2.0, 3.0, 4.0};
  const std::size_t n_rows = 3;
  const std::size_t n_cols = 3;

    ASSERT_THAT(
    [&](){sort::sort_columns_in_column_major_matrix(data, n_rows, n_cols); },
    testing::ThrowsMessage<std::invalid_argument>("sort_columns_in_column_major_matrix - size mismatch")
  );
}

TEST(TestSort, SortColumnsInColumnMajorMatrixAscending) {
  std::vector<double> data = {8861,5031,21,28,281,147,268,198,359,
                              26315,36472,41671,26203,27438,28380,16493,27415,30451,
                              21780,3615,7359,12127,44095.5,41877,42208,42047,8094,
                              30181,31498,20754,15270,8347,23869,19720,28448,1381,
                              3004,33903,519,3572,20083,21148,23731,36082,735};

  std::vector<double> expected_data = {21,28,147,198,268,281,359,5031,8861,
                                      16493,26203,26315,27415,27438,28380,30451,36472,41671,
                                      3615,7359,8094,12127,21780,41877,42047,42208,44095.5,
                                      1381,8347,15270,19720,20754,23869,28448,30181,31498,
                                      519,735,3004,3572,20083,21148,23731,33903,36082};
  const std::size_t n_rows = 9;
  const std::size_t n_cols = 5;

  sort::sort_columns_in_column_major_matrix(data, n_rows, n_cols);

  EXPECT_EQ(data, expected_data);
}
