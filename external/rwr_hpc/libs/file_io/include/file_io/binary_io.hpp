/// @file binary_io.hpp
/// @brief File I/O utilites for read/writing matrices from/to binary format
/// 
/// @author Ken Smith
/// @date 2025-07-24

#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <thread>

/// @namespace file_io
/// @brief Namespace for vector and matrix I/O utility functions
namespace file_io {

/// @brief Load a matrix from binary file into a flat vector.
///
/// @tparam T           The data type of the elements to read (e.g., double, float, int).
///
/// @param file_name    Path to the binary file to read.
/// @param n_rows       Number of rows in the matrix.
/// @param n_cols       Number of columns in the matrix.
///
/// @return A flat vector of size n_rows * n_cols containing the matrix data in row- or column-major order as stored.
///
/// @throws std::runtime_error if the file cannot be opened.
/// @throws std::invalid_argument if n_rows is not positive
/// @throws std::invalid_argument if n_cols is not positive
///
/// @note Assumes the file contains exactly n_rows * n_cols consecutive values of type T.
template <typename T>
std::vector<T> load_binary_block(const std::string& file_name, const std::size_t n_rows, const std::size_t n_cols) {
  std::ifstream fin(file_name, std::ios::binary);
  if (!fin) {
    throw std::runtime_error("load_binary_block - failed to open file: " + file_name);
  }
  if (n_rows == 0) {
    throw std::invalid_argument("load_binary_block - n_rows must be positive");
  }
  if (n_cols == 0) {
    throw std::invalid_argument("load_binary_block - n_cols must be positive");
  }
  std::vector<T> data(n_rows * n_cols);
  fin.read(reinterpret_cast<char*>(data.data()), n_rows * n_cols * sizeof(T));
  return data;
}

/// @brief Asynchronously write a buffer to a binary file on disk.
///
/// @tparam T           The data type of the buffer to write.
///
/// @param file_name    Name of the output file.
/// @param host_buf     Pointer to the data buffer (host memory).
/// @param count        Number of elements of type T to write.
///
/// @details Launches a detached thread to perform the write. If the file cannot be opened,
///          an error is printed to stderr. This function returns immediately and does not guarantee completion before return.
///
/// @note Use with caution in high-concurrency environments to avoid oversubscription or loss of error visibility.
// template <typename T>
// void write_binary_block_to_file_async(std::string file_name, const T* host_buf, std::size_t count) {
//   std::thread([file_name = std::move(file_name), host_buf, count]() {
//     try {
//       std::ofstream fout(file_name, std::ios::binary);
//       if (!fout) {
//         throw std::runtime_error("write_binary_block_to_file_async - failed to open file: " + file_name);
//       }
//       fout.write(reinterpret_cast<const char*>(host_buf), count * sizeof(T));
//       fout.close();
//     } catch (const std::exception& e) {
//       std::cerr << "Async write failed: " << e.what() << '\n';
//     }
//   }).detach();
// }

/// @brief Write a buffer to a binary file on disk (synchronously).
///
/// @tparam T           The data type of the buffer to write.
///
/// @param file_name    Name of the output file.
/// @param host_buf     Pointer to the data buffer (host memory).
/// @param count        Number of elements of type T to write.
////
/// @throws std::runtime_error if the file cannot be opened.
///
/// @note This function blocks until the entire file is written.
template <typename T>
void write_binary_block_to_file(const std::string& file_name, const T* host_buf, std::size_t count) {
  std::ofstream fout(file_name, std::ios::binary);
  if (!fout) {
    throw std::runtime_error("write_binary_block_to_file - failed to open file: " + file_name);
  }
  fout.write(reinterpret_cast<const char*>(host_buf), count * sizeof(T));
  fout.close();
}

/// @brief Scan a directory for binary block files matching the pattern <prefix>_i_j.bin.
///
/// @param directory     The path to the directory to scan.
/// @param file_prefix   The prefix used in the file naming convention (e.g., "corr_block").
///
/// @return A vector of (i, j) index pairs, where each pair corresponds to a file
///         named <prefix>_i_j.bin found in the directory.
///
/// @details This function uses regular expression matching to identify files that
///          match the pattern "<prefix>_i_j.bin", where i and j are integers.
///          The resulting list of (i, j) pairs represents the row and column indices
///          associated with each file.
///
/// @note Only files that exactly match the naming pattern are included.
///       Non-matching files are ignored. The returned vector is not sorted.
std::vector<std::pair<int, int>> find_block_files(const std::string& directory,
                                                  const std::string& file_prefix = "dist_block");

} // namespace file_io
