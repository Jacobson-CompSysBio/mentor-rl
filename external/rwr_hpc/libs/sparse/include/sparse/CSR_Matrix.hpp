/// @file CSR_Matrix.hpp
/// @brief Definition of a compressed sparse row (CSR) matrix class for double-precision values.
///
/// This class provides methods for manipulating sparse matrices in CSR format,
/// including initialization, normalization, row construction, and accessors.
///
/// @author Ken Smith
/// @date 2025-07-24

#pragma once

#include <vector>
#include <string>
#include <cstdint>

/// @class CSR_Matrix
/// @brief A class representing a sparse matrix in compressed sparse row (CSR) format.
///
/// The CSR format stores the matrix as three arrays:
///
///  - `values_`: non-zero entries
///
///  - `col_idx_`: column indices corresponding to each non-zero entry
///
///  - `row_ptr_`: starting indices of each row in `values_` and `col_idx_`
///
/// This class supports column normalization, row-wise insertion, scalar scaling,
/// and other sparse matrix operations commonly used in graph and ML workloads.
class CSR_Matrix {
public:
  std::vector<double> values_;
  std::vector<int32_t> col_idx_;
  std::vector<int32_t> row_ptr_;
  
  unsigned long n_rows_ = 0;
  unsigned long n_cols_ = 0;
  unsigned long nnz_ = 0;

  /// @brief Default constructor. Creates an empty matrix.
  CSR_Matrix() = default;

  /// @brief Construct an empty CSR matrix with given dimensions and reserve space for nnz elements.
  ///
  /// @param n_rows Number of rows.
  /// @param n_cols Number of columns.
  /// @param nnz    Estimated number of non-zero elements.
  ///
  /// @throws std::invalid_argument if nnz > n_rows * n_cols.
  CSR_Matrix(
    unsigned long n_rows,
    unsigned long n_cols,
    unsigned long nnz = 0
  );

  /// @brief Construct a CSR matrix from raw CSR data (values, col_idx, row_ptr).
  ///
  /// @param n_rows   Number of rows.
  /// @param n_cols   Number of columns.
  /// @param values   Non-zero values in the matrix.
  /// @param col_idx  Column indices corresponding to each value.
  /// @param row_ptr  Row pointer array (must have n_rows + 1 elements).
  ///
  /// @throws std::invalid_argument if `values` and `col_idx` vectors are different sizes.
  /// @throws std::invalid_argument if `values` size is greater than `n_rows x n_cols`.
  /// @throws std::invalid_argument if `row_ptr` size does not equal `n_rows + 1`
  /// @throws std::invalid_argument if `row_ptr` does no start at 0 and end at `values.size()`
  /// @throws std::invalid_argument if `row_ptr` values are decreasing
  /// @throws std::invalid_argument if `col_idx` is out of bounds
  /// @throws std::invalid_argument if `col_idx` is not strictly increasing within rows
  CSR_Matrix(
    unsigned long n_rows,
    unsigned long n_cols,
    const std::vector<double> &values,
    const std::vector<int32_t> &col_idx,
    const std::vector<int32_t> &row_ptr
  );

  /// @brief Create a diagonal matrix with the given value along the diagonal.
  ///
  /// @param n     Matrix dimension (square matrix).
  /// @param value Value to place on the diagonal (default: 1.0).
  ///
  /// @return A new CSR_Matrix representing a diagonal matrix.
  static CSR_Matrix  diag(
    unsigned long n,
    double value = 1.0
  );

  /// @brief Initialize the CSR matrix structure with given dimensions and reserve space.
  ///
  /// @param n_rows Number of rows.
  /// @param n_cols Number of columns.
  /// @param nnz    Expected number of non-zero elements.
  ///
  /// @details Resets and prepares internal data structures. All values are cleared.
  ///
  /// @throws std::invalid_argument if nnz > n_rows * n_cols.
  void init(
    unsigned long n_rows,
    unsigned long n_cols,
    unsigned long nnz
  );
  
  /// @brief Normalize the matrix so each column sums to 1.0 (if it contains non-zero values).
  ///
  /// @details For each column with non-zero values, each element is divided by the column sum.
  ///          Columns with no non-zero entries are left unchanged.
  void col_normalize();

  /// @brief Multiply every stored value in the matrix by a scalar.
  ///
  /// @param scale_value The scalar to multiply each non-zero entry by.
  void scale_values(double value);

  /// @brief Add a new row to the matrix with the given column-value pairs.
  ///
  /// @param row              Row index to add.
  /// @param col_value_pairs  Pairs of (column index, value) for the row.
  ///
  /// @details All column indices must be in strictly increasing order.
  ///
  /// @throws std::out_of_range if row or any column index is invalid.
  /// @throws std::runtime_error if the row already contains data.
  /// @throws std::invalid_argument if columns are no strictly increasing within the row
  void add_row(
    std::size_t row,
    const std::vector<std::pair<std::size_t, double>> &col_value_pairs
  );

  /// @brief Add a new row to the matrix using separate value and column vectors.
  ///
  /// @param row     Row index to add.
  /// @param values  Values for the row.
  /// @param cols    Corresponding column indices (must be same length as values).
  ///
  /// @details Column indices must be in strictly increasing order.
  ///
  /// @throws std::out_of_range if row or any column index is invalid.
  /// @throws std::invalid_argument if values.size() != cols.size().
  /// @throws std::runtime_error if the row already contains data.
  /// @throws std::invalid_argument if columns are no strictly increasing within the row
  void add_row(
    std::size_t row,
    const std::vector<double> &_values,
    const std::vector<std::size_t> _cols
  );

  void remove_rows_cols(
    CSR_Matrix& output,
    const std::vector<std::size_t>& rows_to_remove,
    const std::vector<std::size_t>& cols_to_remove
  ) const;

  /// @brief Compute the sum of all elements in each column.
  ///
  /// @return A vector of size n_cols, where each entry is the sum of its column.
  std::vector<double> col_sum() const;

  /// @brief Returns the number of rows in the matrix.
  /// @return The number of rows (n_rows_).
  inline unsigned long n_rows() const { return n_rows_; }

  /// @brief Returns the number of columns in the matrix.
  /// @return The number of columns (n_cols_).
  inline unsigned long n_cols() const { return n_cols_; }

  /// @brief Returns the number of non-zero elements in the matrix.
  /// @return The number of non-zero values (nnz_).
  inline unsigned long nnz() const { return nnz_; }

  /// @brief Returns a const reference to the non-zero values of the matrix.
  /// @return Reference to the values_ vector storing non-zero entries.
  inline const std::vector<double> &get_values() const { return values_; }

  /// @brief Returns a const reference to the column indices of non-zero values.
  /// @return Reference to the col_idx_ vector storing column indices for each value.
  inline const std::vector<int32_t> &get_col_idx() const { return col_idx_; }

  /// @brief Returns a const reference to the row pointer array.
  ///
  /// The row pointer defines the start of each row in the values_ and col_idx_ arrays.
  ///
  /// @return Reference to the row_ptr_ vector.
  inline const std::vector<int32_t> &get_row_ptr() const { return row_ptr_; }
};
