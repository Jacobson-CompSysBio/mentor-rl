#include "gtest/gtest.h"
#include <gmock/gmock-matchers.h>
#include <stdexcept>
#include <sparse/CSR_Matrix.hpp>

TEST(CSR_MatrixTest, CorrectlyInitialized) {
  CSR_Matrix mat;
  mat.init(10, 12, 15);

  EXPECT_EQ(mat.n_rows(), 10);
  EXPECT_EQ(mat.n_cols(), 12);
  EXPECT_EQ(mat.nnz(), 0);

  const auto& values = mat.get_values();
  const auto& col_idx = mat.get_col_idx();
  const auto& row_ptr = mat.get_row_ptr();

  ASSERT_EQ(values.size(), 0);
  ASSERT_EQ(col_idx.size(), 0);
  ASSERT_EQ(row_ptr.size(), 11);
  for (auto r : row_ptr) {
    EXPECT_EQ(r, 0);
  }
}

TEST(CSR_MatrixTest, ThrowsOnTooLargeNnz) {
  ASSERT_THAT(
    [&](){CSR_Matrix mat(10, 5, 1000); },
    testing::ThrowsMessage<std::invalid_argument>("CSR_Matrix::init - nnz cannot be larger than n_rows x n_cols")
  );
}

TEST(CSR_MatrixTest, CorrectlyCreatesDiag) {
  CSR_Matrix mat = CSR_Matrix::diag(7, 1.5);

  EXPECT_EQ(mat.n_rows(), 7);
  EXPECT_EQ(mat.n_cols(), 7);
  EXPECT_EQ(mat.nnz(), 7);

  const auto& values = mat.get_values();
  const auto& col_idx = mat.get_col_idx();
  const auto& row_ptr = mat.get_row_ptr();

  ASSERT_EQ(values.size(), 7);
  ASSERT_EQ(col_idx.size(), 7);
  ASSERT_EQ(row_ptr.size(), 8);

  for (auto v : values) {
    EXPECT_EQ(v, 1.5);
  }

  EXPECT_EQ(col_idx[0], 0);
  EXPECT_EQ(col_idx[1], 1);
  EXPECT_EQ(col_idx[2], 2);
  EXPECT_EQ(col_idx[3], 3);
  EXPECT_EQ(col_idx[4], 4);
  EXPECT_EQ(col_idx[5], 5);
  EXPECT_EQ(col_idx[6], 6);

  EXPECT_EQ(row_ptr[0], 0);
  EXPECT_EQ(row_ptr[1], 1);
  EXPECT_EQ(row_ptr[2], 2);
  EXPECT_EQ(row_ptr[3], 3);
  EXPECT_EQ(row_ptr[4], 4);
  EXPECT_EQ(row_ptr[5], 5);
  EXPECT_EQ(row_ptr[6], 6);
  EXPECT_EQ(row_ptr[7], 7);
}

TEST(CSR_MatrixTest, ConstructorThrowsOnValuesColIdxMismatch) {
  const std::size_t n_rows = 5;
  const std::size_t n_cols = 5;
  const std::vector<double> values = {1.5, 1.7, 2.0, 0.4, 0.6, 0.5, 0.0};
  const std::vector<int32_t> col_idx = {0, 1, 3, 0, 4, 2, 3, 4};
  const std::vector<int32_t> row_ptr = {0, 2, 3, 5, 7, 7};
  ASSERT_THAT(
    [&](){CSR_Matrix mat(n_rows, n_cols, values, col_idx, row_ptr); },
    testing::ThrowsMessage<std::invalid_argument>("CSR_Matrix::CSR_Matrix - values and col_idx must be same size")
  );
}

TEST(CSR_MatrixTest, ConstructorThrowsOnTooLargeNnz) {
  const std::size_t n_rows = 2;
  const std::size_t n_cols = 2;
  const std::vector<double> values = {1.5, 1.7, 2.0, 0.4, 0.6, 0.5, 0.0};
  const std::vector<int32_t> col_idx = {0, 1, 3, 0, 4, 2, 3};
  const std::vector<int32_t> row_ptr = {0, 2, 3};
  ASSERT_THAT(
    [&](){CSR_Matrix mat(n_rows, n_cols, values, col_idx, row_ptr); },
    testing::ThrowsMessage<std::invalid_argument>("CSR_Matrix::CSR_Matrix - nnz cannot be larger than n_rows x n_cols")
  );
}

TEST(CSR_MatrixTest, ConstructorThrowsOnBadRowPtrSize) {
  const std::size_t n_rows = 4;
  const std::size_t n_cols = 5;
  const std::vector<double> values = {1.5, 1.7, 2.0, 0.4, 0.6, 0.5, 0.0};
  const std::vector<int32_t> col_idx = {0, 1, 3, 0, 4, 2, 3};
  const std::vector<int32_t> row_ptr = {0, 2, 3, 5, 7, 7};
  ASSERT_THAT(
    [&](){CSR_Matrix mat(n_rows, n_cols, values, col_idx, row_ptr); },
    testing::ThrowsMessage<std::invalid_argument>("CSR_Matrix::CSR_Matrix - row_ptr.size() must equal n_rows + 1")
  );
}

TEST(CSR_MatrixTest, ConstructorThrowsOnBadRowPtrFirstElement) {
  const std::size_t n_rows = 5;
  const std::size_t n_cols = 5;
  const std::vector<double> values = {1.5, 1.7, 2.0, 0.4, 0.6, 0.5, 0.0};
  const std::vector<int32_t> col_idx = {0, 1, 3, 0, 4, 2, 3};
  const std::vector<int32_t> row_ptr = {1, 2, 3, 5, 7, 7};
  ASSERT_THAT(
    [&](){CSR_Matrix mat(n_rows, n_cols, values, col_idx, row_ptr); },
    testing::ThrowsMessage<std::invalid_argument>("CSR_Matrix::CSR_Matrix - first row_ptr value must be zero")
  );
}

TEST(CSR_MatrixTest, ConstructorThrowsOnBadRowPtrLastElement) {
  const std::size_t n_rows = 5;
  const std::size_t n_cols = 5;
  const std::vector<double> values = {1.5, 1.7, 2.0, 0.4, 0.6, 0.5, 0.0};
  const std::vector<int32_t> col_idx = {0, 1, 3, 0, 4, 2, 3};
  const std::vector<int32_t> row_ptr = {0, 2, 3, 5, 7, 8};
  ASSERT_THAT(
    [&](){CSR_Matrix mat(n_rows, n_cols, values, col_idx, row_ptr); },
    testing::ThrowsMessage<std::invalid_argument>("CSR_Matrix::CSR_Matrix - last row_ptr value must be nnz")
  );
}

TEST(CSR_MatrixTest, ConstructorThrowsOnNonIncreasingRowPtr) {
  const std::size_t n_rows = 5;
  const std::size_t n_cols = 5;
  const std::vector<double> values = {1.5, 1.7, 2.0, 0.4, 0.6, 0.5, 0.0};
  const std::vector<int32_t> col_idx = {0, 1, 3, 0, 4, 2, 3};
  const std::vector<int32_t> row_ptr = {0, 2, 3, 5, 4, 7};
  ASSERT_THAT(
    [&](){CSR_Matrix mat(n_rows, n_cols, values, col_idx, row_ptr); },
    testing::ThrowsMessage<std::invalid_argument>("CSR_Matrix::CSR_Matrix - row_ptr values are decreasing")
  );
}

TEST(CSR_MatrixTest, ConstructorThrowsColIdxOutOfRange) {
  const std::size_t n_rows = 5;
  const std::size_t n_cols = 5;
  const std::vector<double> values = {1.5, 1.7, 2.0, 0.4, 0.6, 0.5, 0.0};
  const std::vector<int32_t> col_idx = {0, 23, 3, 0, 4, 2, 3};
  const std::vector<int32_t> row_ptr = {0, 2, 3, 5, 7, 7};
  ASSERT_THAT(
    [&](){CSR_Matrix mat(n_rows, n_cols, values, col_idx, row_ptr); },
    testing::ThrowsMessage<std::invalid_argument>("CSR_Matrix::CSR_Matrix - col_idx value is out of range")
  );
}

TEST(CSRCSR_MatrixTest_Matrix, ConstructorThrowsColIdxNonIncreasing) {
  const std::size_t n_rows = 5;
  const std::size_t n_cols = 5;
  const std::vector<double> values = {1.5, 1.7, 2.0, 0.4, 0.6, 0.5, 0.0};
  const std::vector<int32_t> col_idx = {2, 0, 3, 0, 4, 2, 3};
  const std::vector<int32_t> row_ptr = {0, 2, 3, 5, 7, 7};
  ASSERT_THAT(
    [&](){CSR_Matrix mat(n_rows, n_cols, values, col_idx, row_ptr); },
    testing::ThrowsMessage<std::invalid_argument>("CSR_Matrix::CSR_Matrix - col_idx values must be inceasing within a row")
  );
}

TEST(CSR_MatrixTest, CorrectlyColumnNormalize) {
  const std::vector<double> values = {1.0,0.0,1.0,1.0,1.5,2.0,0.4};
  const std::vector<int32_t> col_idx = {0,1,2,1,2,1,2};
  const std::vector<int32_t> row_ptr = {0,3,5,7};
  
  CSR_Matrix mat(3, 4, values, col_idx, row_ptr);
  mat.col_normalize();

  const std::vector<double> expected_values = {1.0, 0.0, 1.0/2.9,
                                               1.0/3.0, 1.5/2.9,
                                               2.0/3.0, 0.4/2.9};

  // Check matrix dimension
  EXPECT_EQ(mat.n_rows(), 3);
  EXPECT_EQ(mat.n_cols(), 4);
  // Check number of non-zero elements
  EXPECT_EQ(mat.nnz(), 7);

  // Check values vector
  auto actual_values = mat.get_values();
  ASSERT_EQ(actual_values.size(), expected_values.size());
  for (std::size_t i = 0; i < expected_values.size(); ++i) {
    EXPECT_NEAR(actual_values[i], expected_values[i], 1e-8);
  }
  
  // Check col_idx vector
  auto actual_col_idx = mat.get_col_idx();
  ASSERT_EQ(actual_col_idx.size(), col_idx.size());
  for (std::size_t i = 0; i < col_idx.size(); ++i) {
    EXPECT_EQ(actual_col_idx[i], col_idx[i]);
  }

  // Check row_ptr vector
  auto actual_row_ptr = mat.get_row_ptr();
  ASSERT_EQ(actual_row_ptr.size(), row_ptr.size());
  for (std::size_t i = 0; i < row_ptr.size(); ++i) {
    EXPECT_EQ(actual_row_ptr[i], row_ptr[i]);
  } 
}

TEST(CSR_MatrixTest, CorrectlyScalesValues) {
  const std::vector<double> values = {1.0,2.0,3.0,4.0,1.0,2.0,3.0};
  const std::vector<int32_t> col_idx = {0,1,2,1,2,1,2};
  const std::vector<int32_t> row_ptr = {0,3,5,7};

  CSR_Matrix mat(3, 4, values, col_idx, row_ptr);
  mat.scale_values(2);

  // Check matrix dimension
  EXPECT_EQ(mat.n_rows(), 3);
  EXPECT_EQ(mat.n_cols(), 4);
  // Check number of non-zero elements
  EXPECT_EQ(mat.nnz(), 7);

  // Check values vector
  auto actual_values = mat.get_values();
  ASSERT_EQ(actual_values.size(), values.size());
  for (std::size_t i = 0; i < values.size(); ++i) {
    EXPECT_NEAR(actual_values[i], 2.0 * values[i], 1e-8);
  }
  
  // Check col_idx vector
  auto actual_col_idx = mat.get_col_idx();
  ASSERT_EQ(actual_col_idx.size(), col_idx.size());
  for (std::size_t i = 0; i < col_idx.size(); ++i) {
    EXPECT_EQ(actual_col_idx[i], col_idx[i]);
  }

  // Check row_ptr vector
  auto actual_row_ptr = mat.get_row_ptr();
  ASSERT_EQ(actual_row_ptr.size(), row_ptr.size());
  for (std::size_t i = 0; i < row_ptr.size(); ++i) {
    EXPECT_EQ(actual_row_ptr[i], row_ptr[i]);
  } 
}

TEST(CSR_MatrixTest, AddRowThrowsIfRowOutOfRange) {
  CSR_Matrix mat(5,5,3);
  std::vector<std::pair<std::size_t, double>> col_value_pairs;
  col_value_pairs.push_back(std::make_pair(0, 1.1));
  col_value_pairs.push_back(std::make_pair(1, 1.2));

  ASSERT_THAT(
    [&](){mat.add_row(7, col_value_pairs); },
    testing::ThrowsMessage<std::out_of_range>("CSR_Matrix::add_row - row is out of range")
  );
}

TEST(CSR_MatrixTest, AddRowsCorrectly) {
  CSR_Matrix mat(5,5,2);
  std::vector<std::pair<std::size_t, double>> col_value_pairs;
  col_value_pairs.push_back(std::make_pair(0, 1.1));
  col_value_pairs.push_back(std::make_pair(1, 1.2));

  mat.add_row(1, col_value_pairs);

  EXPECT_EQ(mat.n_rows(), 5);
  EXPECT_EQ(mat.n_cols(), 5);
  EXPECT_EQ(mat.nnz(), 2);
  
  auto actual_values = mat.get_values();
  auto actual_col_idx = mat.get_col_idx();
  auto actual_row_ptr = mat.get_row_ptr();

  ASSERT_EQ(actual_values.size(), 2);
  EXPECT_EQ(actual_values[0], 1.1);
  EXPECT_EQ(actual_values[1], 1.2);

  ASSERT_EQ(actual_col_idx.size(), 2);
  EXPECT_EQ(actual_col_idx[0], 0);
  EXPECT_EQ(actual_col_idx[1], 1);

  ASSERT_EQ(actual_row_ptr.size(), 6);
  EXPECT_EQ(actual_row_ptr[0], 0);
  EXPECT_EQ(actual_row_ptr[1], 0);
  EXPECT_EQ(actual_row_ptr[2], 2);
  EXPECT_EQ(actual_row_ptr[3], 2);
  EXPECT_EQ(actual_row_ptr[4], 2);
  EXPECT_EQ(actual_row_ptr[5], 2);
}

TEST(CSR_MatrixTest, AddRowThrowsIfMatrixHasDataInRow) {
  CSR_Matrix mat(5,5,3);
  std::vector<std::pair<std::size_t, double>> col_value_pairs;
  col_value_pairs.push_back(std::make_pair(0, 1.1));
  col_value_pairs.push_back(std::make_pair(1, 1.2));

  mat.add_row(1, col_value_pairs);

  col_value_pairs.clear();
  col_value_pairs.push_back(std::make_pair(2, 1.1));
  col_value_pairs.push_back(std::make_pair(3, 1.2));

  ASSERT_THAT(
    [&](){mat.add_row(1, col_value_pairs); },
    testing::ThrowsMessage<std::runtime_error>("CSR_Matrix::add_row - matrix already has data at or after row")
  );
}

TEST(CSR_MatrixTest, AddRowThrowsIfMatrixHasDataInLaterRow) {
  CSR_Matrix mat(5,5,3);
  std::vector<std::pair<std::size_t, double>> col_value_pairs;
  col_value_pairs.push_back(std::make_pair(0, 1.1));
  col_value_pairs.push_back(std::make_pair(1, 1.2));

  mat.add_row(1, col_value_pairs);

  col_value_pairs.clear();
  col_value_pairs.push_back(std::make_pair(2, 1.1));
  col_value_pairs.push_back(std::make_pair(3, 1.2));

  ASSERT_THAT(
    [&](){mat.add_row(0, col_value_pairs); },
    testing::ThrowsMessage<std::runtime_error>("CSR_Matrix::add_row - matrix already has data at or after row")
  );
}

TEST(CSR_MatrixTest, AddRowThrowsIfColumnOutOfRange) {
  CSR_Matrix mat(5,5,3);
  std::vector<std::pair<std::size_t, double>> col_value_pairs;
  col_value_pairs.push_back(std::make_pair(1, 1.1));
  col_value_pairs.push_back(std::make_pair(19, 1.2));

  ASSERT_THAT(
    [&](){mat.add_row(1, col_value_pairs); },
    testing::ThrowsMessage<std::out_of_range>("CSR_Matrix::add_row - column is out of range")
  );
}

TEST(CSR_MatrixTest, AddRowsThrowsIfColumnsNotIncreasing) {
  CSR_Matrix mat(5,5,3);
  std::vector<std::pair<std::size_t, double>> col_value_pairs;
  col_value_pairs.push_back(std::make_pair(1, 1.1));
  col_value_pairs.push_back(std::make_pair(0, 1.2));

  ASSERT_THAT(
    [&](){mat.add_row(1, col_value_pairs); },
    testing::ThrowsMessage<std::invalid_argument>("CSR_Matrix::add_row - columns need to added in increasing order")
  );
}

TEST(CSR_MatrixTest, AddRowV2ThrowsIfRowOutOfRange) {
  CSR_Matrix mat(5,5,3);
  std::vector<double> values = {1.1, 1.2};
  std::vector<std::size_t> col_idx = {0, 1};

  ASSERT_THAT(
    [&](){mat.add_row(7, values, col_idx); },
    testing::ThrowsMessage<std::out_of_range>("CSR_Matrix::add_row - row is out of range")
  );
}

TEST(CSR_MatrixTest, AddRowsV2Correctly) {
  CSR_Matrix mat(5,5,2);
  std::vector<double> values = {1.1, 1.2};
  std::vector<std::size_t> col_idx = {0, 1};

  mat.add_row(1, values, col_idx);

  EXPECT_EQ(mat.n_rows(), 5);
  EXPECT_EQ(mat.n_cols(), 5);
  EXPECT_EQ(mat.nnz(), 2);
  
  auto actual_values = mat.get_values();
  auto actual_col_idx = mat.get_col_idx();
  auto actual_row_ptr = mat.get_row_ptr();

  ASSERT_EQ(actual_values.size(), 2);
  EXPECT_EQ(actual_values[0], 1.1);
  EXPECT_EQ(actual_values[1], 1.2);

  ASSERT_EQ(actual_col_idx.size(), 2);
  EXPECT_EQ(actual_col_idx[0], 0);
  EXPECT_EQ(actual_col_idx[1], 1);

  ASSERT_EQ(actual_row_ptr.size(), 6);
  EXPECT_EQ(actual_row_ptr[0], 0);
  EXPECT_EQ(actual_row_ptr[1], 0);
  EXPECT_EQ(actual_row_ptr[2], 2);
  EXPECT_EQ(actual_row_ptr[3], 2);
  EXPECT_EQ(actual_row_ptr[4], 2);
  EXPECT_EQ(actual_row_ptr[5], 2);
}

TEST(CSR_MatrixTest, AddRowV2ThrowsIfMatrixHasDataInRow) {
  CSR_Matrix mat(5,5,3);
  std::vector<double> values = {1.1, 1.2};
  std::vector<std::size_t> col_idx = {0, 1};
  
  mat.add_row(1, values, col_idx);

  col_idx = {2, 3};

  ASSERT_THAT(
    [&](){mat.add_row(1, values, col_idx); },
    testing::ThrowsMessage<std::runtime_error>("CSR_Matrix::add_row - matrix already has data at or after row")
  );
}

TEST(CSR_MatrixTest, AddRowV2ThrowsIfMatrixHasDataInLaterRow) {
  CSR_Matrix mat(5,5,3);
  std::vector<double> values = {1.1, 1.2};
  std::vector<std::size_t> col_idx = {0, 1};
  mat.add_row(1, values, col_idx);

  ASSERT_THAT(
    [&](){mat.add_row(0, values, col_idx); },
    testing::ThrowsMessage<std::runtime_error>("CSR_Matrix::add_row - matrix already has data at or after row")
  );
}

TEST(CSR_MatrixTest, AddRowV2ThrowsIfColumnOutOfRange) {
  CSR_Matrix mat(5,5,3);
  std::vector<double> values = {1.1, 1.2};
  std::vector<std::size_t> col_idx = {1, 19};
  
  ASSERT_THAT(
    [&](){mat.add_row(1, values, col_idx); },
    testing::ThrowsMessage<std::out_of_range>("CSR_Matrix::add_row - column is out of range")
  );
}

TEST(CSR_MatrixTest, AddRowsV2IfColumnsNotIncreasing) {
  CSR_Matrix mat(5,5,3);
  std::vector<double> values = {1.1, 1.2};
  std::vector<std::size_t> col_idx = {1, 0};

  ASSERT_THAT(
    [&](){mat.add_row(1, values, col_idx); },
    testing::ThrowsMessage<std::runtime_error>("CSR_Matrix::add_row - columns need to added in increasing order")
  );
}

TEST(CSR_MatrixTest, RemoveRowsFromMatrix) {
  const std::size_t n_rows = 12;
  const std::size_t n_cols = 12;
  const std::vector<int32_t> row_ptr {0,2,3,5,6,7,9,10,12,14,16,18,20};
  const std::vector<int32_t> col_idx = {1,2,0,0,1,1,6,4,6,5,5,6,9,10,8,10,8,11,8,10};
  const std::vector<double> values = {1.0/3.0, 1.0,
                                      1.0/2.0,
                                      1.0/2.0, 1.0/3.0,
                                      1.0/3.0,
                                      1.0/4.0,
                                      1.0, 1.0/4.0,
                                      1.0/2.0,
                                      1.0/2.0, 1.0/2.0,
                                      1.0, 1.0,
                                      1.0/3.0, 2.0,
                                      1.0/3.0, 2.0,
                                      1.0/3.0, 3.0};

  CSR_Matrix in(n_rows, n_cols, values, col_idx, row_ptr);

  CSR_Matrix out;
  const std::vector<std::size_t> rows_to_removes = {3,4,5};
  const std::vector<std::size_t> cols_to_removes;
  in.remove_rows_cols(out, rows_to_removes, cols_to_removes);

  EXPECT_EQ(out.n_rows(), 9);
  EXPECT_EQ(out.n_cols(), 12);
  EXPECT_EQ(out.nnz(), 16);

  const std::vector<int32_t> expected_row_ptr = {0,2,3,5,6,8,10,12,14,16};
  const std::vector<int32_t> expected_col_idx = {1,2,0,0,1,5,5,6,9,10,8,10,8,11,8,10};
  const std::vector<double> expected_values = {1.0/3.0, 1.0,
                                               1.0/2.0,
                                               1.0/2.0, 1.0/3.0,
                                               1.0/2.0,
                                               1.0/2.0, 1.0/2.0,
                                               1.0, 1.0,
                                               1.0/3.0, 2.0,
                                               1.0/3.0, 2.0,
                                               1.0/3.0, 3.0};
  
  EXPECT_EQ(expected_row_ptr, out.row_ptr_);
  EXPECT_EQ(expected_col_idx, out.col_idx_);
  EXPECT_EQ(expected_values, out.values_);
}

TEST(CSR_MatrixTest, RemoveColsFromMatrix) {
  const std::size_t n_rows = 12;
  const std::size_t n_cols = 12;
  const std::vector<int32_t> row_ptr {0,2,3,5,6,7,9,10,12,14,16,18,20};
  const std::vector<int32_t> col_idx = {1,2,0,0,1,1,6,4,6,5,5,6,9,10,8,10,8,11,8,10};
  const std::vector<double> values = {1.0/3.0, 1.0,
                                      1.0/2.0,
                                      1.0/2.0, 1.0/3.0,
                                      1.0/3.0,
                                      1.0/4.0,
                                      1.0, 1.0/4.0,
                                      1.0/2.0,
                                      1.0/2.0, 1.0/2.0,
                                      1.0, 1.0,
                                      1.0/3.0, 2.0,
                                      1.0/3.0, 2.0,
                                      1.0/3.0, 3.0};

  CSR_Matrix in(n_rows, n_cols, values, col_idx, row_ptr);

  CSR_Matrix out;
  const std::vector<std::size_t> rows_to_removes;
  const std::vector<std::size_t> cols_to_removes = {3,4,5};
  in.remove_rows_cols(out, rows_to_removes, cols_to_removes);

  EXPECT_EQ(out.n_rows(), 12);
  EXPECT_EQ(out.n_cols(), 9);
  EXPECT_EQ(out.nnz(), 17);

  const std::vector<int32_t> expected_row_ptr = {0,2,3,5,6,7,8,8,9,11,13,15,17};
  const std::vector<int32_t> expected_col_idx = {1,2,0,0,1,1,3,3,3,6,7,5,7,5,8,5,7};
  const std::vector<double> expected_values = {1.0/3.0, 1.0,
                                               1.0/2.0,
                                               1.0/2.0, 1.0/3.0,
                                               1.0/3.0,
                                               1.0/4.0,
                                               1.0/4.0,
                                               1.0/2.0,
                                               1.0, 1.0,
                                               1.0/3.0, 2.0,
                                               1.0/3.0, 2.0,
                                               1.0/3.0, 3.0};
  
  EXPECT_EQ(expected_row_ptr, out.row_ptr_);
  EXPECT_EQ(expected_col_idx, out.col_idx_);
  EXPECT_EQ(expected_values, out.values_);
}

TEST(CSR_MatrixTest, RemoveRowsColsFromMatrix) {
  const std::size_t n_rows = 12;
  const std::size_t n_cols = 12;
  const std::vector<int32_t> row_ptr {0,2,3,5,6,7,9,10,12,14,16,18,20};
  const std::vector<int32_t> col_idx = {1,2,0,0,1,1,6,4,6,5,5,6,9,10,8,10,8,11,8,10};
  const std::vector<double> values = {1.0/3.0, 1.0,
                                      1.0/2.0,
                                      1.0/2.0, 1.0/3.0,
                                      1.0/3.0,
                                      1.0/4.0,
                                      1.0, 1.0/4.0,
                                      1.0/2.0,
                                      1.0/2.0, 1.0/2.0,
                                      1.0, 1.0,
                                      1.0/3.0, 2.0,
                                      1.0/3.0, 2.0,
                                      1.0/3.0, 3.0};

  CSR_Matrix in(n_rows, n_cols, values, col_idx, row_ptr);

  CSR_Matrix out;
  const std::vector<std::size_t> rows_to_removes = {3,4,5};
  const std::vector<std::size_t> cols_to_removes = {3,4,5};
  in.remove_rows_cols(out, rows_to_removes, cols_to_removes);

  EXPECT_EQ(out.n_rows(), 9);
  EXPECT_EQ(out.n_cols(), 9);
  EXPECT_EQ(out.nnz(), 14);

  const std::vector<int32_t> expected_row_ptr = {0,2,3,5,5,6,8,10,12,14};
  const std::vector<int32_t> expected_col_idx = {1,2,0,0,1,3,6,7,5,7,5,8,5,7};
  const std::vector<double> expected_values = {1.0/3.0, 1.0,
                                               1.0/2.0,
                                               1.0/2.0, 1.0/3.0,
                                               1.0/2.0,
                                               1.0, 1.0,
                                               1.0/3.0, 2.0,
                                               1.0/3.0, 2.0,
                                               1.0/3.0, 3.0};
  
  EXPECT_EQ(expected_row_ptr, out.row_ptr_);
  EXPECT_EQ(expected_col_idx, out.col_idx_);
  EXPECT_EQ(expected_values, out.values_);
}

TEST(CSR_MatrixTest, CorrectlyCalculateColumnSum) {
  const std::vector<double> values = {1.0,0.0,1.0,1.0,1.5,2.0,0.4};
  const std::vector<int32_t> col_idx = {0,1,2,1,2,1,2};
  const std::vector<int32_t> row_ptr = {0,3,5,7};
  
  CSR_Matrix mat(3, 4, values, col_idx, row_ptr);

  const auto col_sum = mat.col_sum();

  ASSERT_EQ(col_sum.size(), 4);
  EXPECT_NEAR(col_sum[0], 1.0, 1e-8);
  EXPECT_NEAR(col_sum[1], 3.0, 1e-8);
  EXPECT_NEAR(col_sum[2], 2.9, 1e-8);
  EXPECT_NEAR(col_sum[3], 0.0, 1e-8);
}
