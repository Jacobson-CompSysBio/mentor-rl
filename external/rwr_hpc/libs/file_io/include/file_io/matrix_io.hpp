/// @file matrix_io.hpp
/// @brief File I/O utilities for writting collumn-major matrices
/// 
/// @author Ken Smith
/// @date 2025-07-24

#pragma once

#include <string>
#include <vector>

/// @namespace file_io
/// @brief Namespace for vector and matrix I/O utility functions
namespace file_io {

/// \enum DIST_MATRIX_MODE
/// \brief Specifies which part of a distance matrix to print.
///
/// Used by functions such as file_io::print_column_major_distance_matrix
/// to control whether the full matrix, upper triangle, or lower triangle
/// is written.
///
/// - ALL:    Print the full matrix.
//
/// - UPPER:  Print only the upper triangle (elements where col > row).
//
/// - LOWER:  Print only the lower triangle (elements where col < row).
enum class DIST_MATRIX_MODE {
  ALL,   ///< Print the full matrix.
  UPPER, ///< Print only the upper triangle.
  LOWER  ///< Print only the lower triangle.
};

/// @brief Write a column-major matrix of doubles to a text file in tab-separated format.
///
/// @param file_name     Path to the output text file.
/// @param matrix        Flattened column-major matrix of size n_rows × n_cols.
/// @param n_rows        Number of rows in the matrix.
/// @param n_cols        Number of columns in the matrix.
/// @param row_labels    Optional row labels (must match n_rows if provided) (default: {}).
/// @param col_labels    Optional column labels (must match n_cols if provided) (default: {}).
/// @param precision     Number of decimal places to print (default: 3).
/// @param scientific    Whether to use scientific notation (default: false).
/// @param transpose     If true, print the transpose of the matrix (default: false).
///
/// @throws std::invalid_argument if the matrix size or label dimensions are inconsistent.
/// @throws std::runtime_error if the file cannot be opened.
void print_column_major_matrix(
  const std::string& file_name,
  const std::vector<double>& matrix,
  const std::size_t n_rows,
  const std::size_t n_cols,
  const std::vector<std::string>& row_labels = {},
  const std::vector<std::string>& col_labels = {},
  const int precision = 3,
  const bool scientific = false,
  const bool transpose = false
);

/// @brief Write a column-major matrix of bools to a text file in tab-separated format.
///
/// @param file_name     Path to the output text file.
/// @param matrix        Flattened column-major matrix of size n_rows × n_cols.
/// @param n_rows        Number of rows in the matrix.
/// @param n_cols        Number of columns in the matrix.
/// @param row_labels    Optional row labels (must match n_rows if provided) (default: false).
/// @param col_labels    Optional column labels (must match n_cols if provided) (default: false).
/// @param transpose     If true, print the transpose of the matrix (default: false).
///
/// @throws std::invalid_argument if the matrix size or label dimensions are inconsistent.
/// @throws std::runtime_error if the file cannot be opened.
void print_column_major_matrix(
  const std::string& file_name,
  const std::vector<bool>& matrix,
  const std::size_t n_rows,
  const std::size_t n_cols,
  const std::vector<std::string>& row_labels = {},
  const std::vector<std::string>& col_labels = {},
  const bool transpose = false
);

/// @brief Write a column-major matrix of distances to a text file with optional filtering.
///
/// @param file_name     Path to the output file.
/// @param matrix        Flattened column-major matrix of size n_rows × n_cols.
/// @param dim           Dimension of the full square matrix.
/// @param n_cols        Number of columns.
/// @param labels        Optional labels (size must match dim if provided) (default: {}).
/// @param precision     Decimal precision for printed values (default: 3).
/// @param scientific    Whether to print in scientific notation (default: false).
/// @param mode          Output mode: ALL, UPPER, or LOWER (as defined in DIST_MATRIX_MODE) (default: ALL).
///
/// @throws std::invalid_argument if matrix size or label dimensions are inconsistent.
/// @throws std::runtime_error if the file cannot be opened.
void print_column_major_distance_matrix(
  const std::string& file_name,
  const std::vector<double>& matrix,
  const std::size_t dim,
  const std::vector<std::string>& labels = {},
  const int precision = 3,
  const bool scientific = false,
  const DIST_MATRIX_MODE mode = DIST_MATRIX_MODE::ALL
);

// /// @brief Print the lower triangle of a symmetric matrix given its upper triangle (flattened).
// ///
// /// @param file_name       Output file name.
// /// @param upper_tri       Flattened upper triangle of the matrix (column-major).
// /// @param dim             Dimension of the full square matrix.
// /// @param row_labels      Optional row labels (size must match dim if provided) (default: {}).
// /// @param col_labels      Optional column labels (size must match dim if provided) (default: {}).
// /// @param print_diagonal  Whether to include diagonal values (default: true).
// /// @param na_token        Token to print for undefined entries (default: "NA").
// /// @param precision       Decimal precision for printed values (default: 3).
// ///
// /// @throws std::invalid_argument if matrix size or label dimensions are inconsistent.
// /// @throws std::runtime_error if the file cannot be opened.
// void print_lower_triangle_from_upper(
//   const std::string& file_name,
//   const std::vector<double>& upper_tri,
//   const std::size_t dim,
//   const std::vector<std::string>& row_labels = {},
//   const std::vector<std::string>& col_labels = {},
//   const bool print_diagonal = true,
//   const std::string& na_token = "NA",
//   const int precision = 3
// );

/// @brief Load and combine binary matrix blocks into a single flattened column-major matrix.
///
/// @param output_matrix   Output buffer for the combined matrix. Will be resized to N×N, where N = sum(N_ranks).
/// @param directory       Directory containing the binary block files.
/// @param file_prefix     Prefix used in the block file names (e.g., \"corr_block\" for files like corr_block_0_1.bin).
/// @param N_ranks         Vector indicating the number of rows per block (per rank/layer).
/// @details Each block file is assumed to be named as <file_prefix>_i_j.bin and stored in column-major order.
///          This function places each block into the correct location in the output matrix.
///
/// @throws std::runtime_error if a block file cannot be opened.
/// @throws std::invalid_argument if any block size is inconsistent with N_ranks.
void read_matrix_from_binary_blocks(
  std::vector<double>& output_matrix,
  const std::string& dir,
  const std::string& file_prefix,
  const std::vector<std::size_t>& N_ranks
);

void write_column_major_matrix_with_yaml(
  const std::string& bin_file,
  const std::string& yaml_file,
  const std::vector<double>& matrix,
  std::size_t n_rows,
  std::size_t n_cols,
  const std::vector<std::string>& row_labels = {},
  const std::vector<std::string>& col_labels = {},
  int precision = 6,
  bool scientific = false,
  bool transpose = false
);

} // namespace file_io
