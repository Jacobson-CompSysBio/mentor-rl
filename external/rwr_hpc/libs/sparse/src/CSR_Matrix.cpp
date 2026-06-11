#include "sparse/CSR_Matrix.hpp"
#include <numeric>  // for std::iota
#include <stdexcept>
#include <algorithm>
#include <unordered_set>
#include <map>

CSR_Matrix::CSR_Matrix(
  unsigned long n_rows,
  unsigned long n_cols,
  const unsigned long nnz
) {
  init(n_rows, n_cols, nnz);
}

CSR_Matrix::CSR_Matrix(
  unsigned long n_rows,
  unsigned long n_cols,
  const std::vector<double> &values,
  const std::vector<int32_t> &col_idx,
  const std::vector<int32_t> &row_ptr
) {
  // Check that values.size() == col_idx.size()
  const unsigned long nnz = values.size();
  if (nnz != col_idx.size()) {
    throw std::invalid_argument("CSR_Matrix::CSR_Matrix - values and col_idx must be same size");
  }
  // Check that nnz fits in matrix
  if (nnz > n_rows * n_cols) {
    throw std::invalid_argument("CSR_Matrix::CSR_Matrix - nnz cannot be larger than n_rows x n_cols");
  }
  // Check that row_ptr size matches expected value
  if (row_ptr.size() != n_rows + 1) {
    throw std::invalid_argument("CSR_Matrix::CSR_Matrix - row_ptr.size() must equal n_rows + 1");
  }
  // Check that row_ptr values are valid
  if (row_ptr[0] != 0) {
    throw std::invalid_argument("CSR_Matrix::CSR_Matrix - first row_ptr value must be zero");
  }
  if (row_ptr[n_rows] != nnz) {
    throw std::invalid_argument("CSR_Matrix::CSR_Matrix - last row_ptr value must be nnz");
  }
  for (std::size_t i = 0; i < n_rows; ++i) {
    if (row_ptr[i] > row_ptr[i + 1]) {
      throw std::invalid_argument("CSR_Matrix::CSR_Matrix - row_ptr values are decreasing");
    }
  }

  // Check the col_idx values
  for (std::size_t i = 0; i < n_rows; ++i) {
    for (int32_t k = row_ptr[i]; k < row_ptr[i+1]; ++k) {
      if (col_idx[k] >= n_cols) {
        throw std::invalid_argument("CSR_Matrix::CSR_Matrix - col_idx value is out of range");
      }
    }
    for (int32_t k = row_ptr[i]; k < row_ptr[i+1]-1; ++k) {
      if (col_idx[k] >= col_idx[k+1]) {
        throw std::invalid_argument("CSR_Matrix::CSR_Matrix - col_idx values must be inceasing within a row");
      }
    }
  }

  n_rows_ = n_rows;
  n_cols_ = n_cols;
  nnz_ = nnz;

  values_ = values;
  col_idx_ = col_idx;
  row_ptr_ = row_ptr;
}

CSR_Matrix CSR_Matrix::diag(
  unsigned long n,
  double value
) {
  CSR_Matrix mat;
  mat.init(n, n, n);
  
  for (std::size_t i = 0; i < n; ++i) {
    mat.values_.push_back(value);
    mat.col_idx_.push_back(static_cast<int32_t>(i));
    mat.row_ptr_[i+1] = static_cast<int32_t>(i+1);
  }

  mat.nnz_ = mat.values_.size();

  return mat;
}

void CSR_Matrix::init(
  unsigned long n_rows,
  unsigned long n_cols,
  unsigned long nnz
) {
  // Check that nnz can fit in matrix
  if (nnz > n_rows * n_cols) {
    throw std::invalid_argument("CSR_Matrix::init - nnz cannot be larger than n_rows x n_cols");
  }

  // Set matrix dim
  n_rows_ = n_rows;
  n_cols_ = n_cols;

  // Set nnz
  nnz_ = 0;

  // Reserve space for vectors
  values_.clear();
  values_.reserve(nnz_);
  col_idx_.clear();
  col_idx_.reserve(nnz_);
  row_ptr_.clear();
  row_ptr_.resize(n_rows_ + 1, 0);
}

void CSR_Matrix::col_normalize() {
  std::vector<double> col_sums(n_cols_, 0.0);

  // Calculate the sum of each column 
  for (std::size_t k = 0; k < nnz_; ++k) {
    col_sums[col_idx_[k]] += values_[k];
  }

  // Normalize the values in each column to sum to 1
  for (std::size_t k = 0; k < nnz_; ++k) {
    if (col_sums[col_idx_[k]] != 0.0) {
      values_[k] /= col_sums[col_idx_[k]];
    }
  }
}

void CSR_Matrix::scale_values(double scale_value) {
  for (auto& v : values_) {
    v *= scale_value;
  }
}

void CSR_Matrix::add_row(
  std::size_t row,
  const std::vector<std::pair<std::size_t, double>> &col_value_pairs
) {
  // Throw if row is out of range
  if (row >= n_rows_) {
    throw std::out_of_range("CSR_Matrix::add_row - row is out of range");
  }
  // Throw exception if matrix has rows at or after input
  const int32_t row_ptr_value = row_ptr_[row];
  for (std::size_t i = row; i <= n_rows_; ++i) {
    if (row_ptr_[i] != row_ptr_value) {
      throw std::runtime_error("CSR_Matrix::add_row - matrix already has data at or after row");
    }
  }

  // Add col_idx and values
  bool first_element = true;
  std::size_t latest_col = 0;
  for (const auto &p: col_value_pairs) {
    if (p.first >= n_cols_) {
      throw std::out_of_range("CSR_Matrix::add_row - column is out of range");
    }
    col_idx_.push_back(static_cast<int32_t>(p.first));
    values_.push_back(p.second);

    // If first element, update threshold and flag
    if (first_element) {
      latest_col = p.first;
      first_element = !first_element;
    } else {
      // Throw an error if columns are not increasing, else update threshold
      if (p.first <= latest_col) {
        throw std::invalid_argument("CSR_Matrix::add_row - columns need to added in increasing order");
      } else {
        latest_col = p.first;
      }
    }
  }

  // Update row ptr values to account for new edges
  for (std::size_t i = row + 1; i < row_ptr_.size(); ++i) {
    row_ptr_[i] += col_value_pairs.size();
  }

  // Update nnz
  nnz_ += col_value_pairs.size();
}

void CSR_Matrix::add_row(
  std::size_t row,
  const std::vector<double> &_values,
  const std::vector<std::size_t> _cols
) {
  // Throw if row is out of range
  if (row >= n_rows_) {
    throw std::out_of_range("CSR_Matrix::add_row - row is out of range");
  }
  // Thro is _values.size() != _cols.size()
  if (_values.size() != _cols.size()) {
    throw std::invalid_argument("CSR_Matrix::add_row - _values and _cols vectors must be the same size");
  }
  // Throw exception if matrix has rows at or after input
  const std::size_t row_ptr_value = row_ptr_[row];
  for (std::size_t i = row; i <= n_rows_; ++i) {
    if (row_ptr_[i] != row_ptr_value) {
      throw std::runtime_error("CSR_Matrix::add_row - matrix already has data at or after row");
    }
  }

  // Add col_idx and values
  bool first_element = true;
  std::size_t latest_col = 0;
  for (std::size_t k = 0; k < _values.size(); ++k) {
    if (_cols[k] >= n_cols_) {
      throw std::out_of_range("CSR_Matrix::add_row - column is out of range");
    }
    col_idx_.push_back(_cols[k]);
    values_.push_back(_values[k]);

    // If first element, update threshold and flag
    if (first_element) {
      latest_col = _cols[k];
      first_element = !first_element;
    } else {
      // Throw an error if columns are not increasing, else update threshold
      if (_cols[k] <= latest_col) {
        throw std::runtime_error("CSR_Matrix::add_row - columns need to added in increasing order");
      } else {
        latest_col = _cols[k];
      }
    }
  }

  // Update row ptr values to account for new edges
  for (std::size_t i = row + 1; i < row_ptr_.size(); ++i) {
    row_ptr_[i] += _values.size();
  }

  // Update nnz
  nnz_ += _values.size();
}

void CSR_Matrix::remove_rows_cols(
  CSR_Matrix& output,
  const std::vector<std::size_t>& rows_to_remove,
  const std::vector<std::size_t>& cols_to_remove
) const {
  // Clear any existing data and reserve space equal to current matrix size
  output.col_idx_.clear();
  output.row_ptr_.clear();
  output.values_.clear();
  output.col_idx_.reserve(this->col_idx_.size());
  output.row_ptr_.reserve(this->row_ptr_.size());
  output.values_.reserve(this->values_.size());

  // Create hash for rows and cols to remove
  std::unordered_set<std::size_t> rows_to_remove_set, cols_to_remove_set;
  for (auto i : rows_to_remove) {
    rows_to_remove_set.insert(i);
  }
  for (auto j : cols_to_remove) {
    cols_to_remove_set.insert(j);
  }

  // Create conversion from the original column index to new columns index
  std::map<std::size_t, std::size_t> new_col_idx;
  std::size_t new_idx = 0;
  for (std::size_t j = 0; j < this->n_cols(); ++j) {
    if (cols_to_remove.empty() || cols_to_remove_set.find(j) == cols_to_remove_set.end()) {
      new_col_idx.emplace(j, new_idx);
      ++new_idx;
    }
  }

  output.row_ptr_.push_back(0);
  // Loop over each row
  for (std::size_t i = 0; i < this->n_rows(); ++i) {
    // If rows_to_remove is not empty, check that the current row is not supposed to be removed
    if (rows_to_remove.empty() || rows_to_remove_set.find(i) == rows_to_remove_set.end()) {
      // Loop through all values in current row
      std::size_t num_cols_in_row = 0;
      for (std::size_t k = this->row_ptr_[i]; k < this->row_ptr_[i+1]; ++k) {
        std::size_t j = this->col_idx_[k];

        // Check if column is in the map
        if (new_col_idx.find(j) != new_col_idx.end()) {
          output.col_idx_.push_back(new_col_idx[j]);
          output.values_.push_back(this->values_[k]);
          ++num_cols_in_row;
        }
      }

      output.row_ptr_.push_back(num_cols_in_row + output.row_ptr_[output.row_ptr_.size()-1]);
    }
  }

  output.n_rows_ = this->n_rows_ - rows_to_remove.size();
  output.n_cols_ = this->n_cols_ - cols_to_remove.size();
  output.nnz_ = output.values_.size();
}

std::vector<double> CSR_Matrix::col_sum() const {
  std::vector<double> sum(n_cols_, 0.0);

  for (std::size_t k = 0; k < values_.size(); ++k) {
    sum[col_idx_[k]] += values_[k];
  }

  return sum;
}
