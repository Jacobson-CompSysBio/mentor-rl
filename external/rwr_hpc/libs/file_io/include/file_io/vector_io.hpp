/// @file vector_io.hpp
/// @brief File I/O utilites for reading and writing 1D vectors to/from text files
///
/// This header provides generic template functions for printing and reading vectors,
/// with support for formatting, column vs. row orientation, and precision control
/// for floating-point types.
/// 
/// @author Ken Smith
/// @date 2025-07-24

#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <type_traits>

/// @namespace file_io
/// @brief Namespace for vector and matrix I/O utility functions
namespace file_io {

/// @brief Write a 1D vector to a text file, optionally formatted as a column or row.
///
/// @tparam T            The data type of the elements (e.g., int, double, std::string).
///
/// @param data          The vector to be written.
/// @param file_name     The output file path.
/// @param column_vec    If true (default), writes one element per line; otherwise, all elements on one line, delimited.
/// @param precision     Number of decimal places to print (only for floating-point types).
/// @param scientific    Whether to use scientific notation (only for floating-point types).
/// @param delim         Delimiter to use between elements if column_vec is false (default: tab).
///
/// @throws std::runtime_error if the file cannot be opened.
///
/// @details For floating-point types, the numeric output format and precision are controlled by the `scientific` and `precision` parameters.
///          For non-floating types, `precision` and `scientific` are ignored.
template<typename T>
void print_vector(const std::vector<T>& data,
                  const std::string &file_name,
                  const bool column_vec = true,
                  const int precision = 3,
                  const bool scientific = false,
                  const char delim = '\t') {
  std::ofstream out(file_name);
  if (!out.is_open()) {
    throw std::runtime_error("file_io::print_vector - unable to open file " + file_name);
  }

  // Set numeric formatting for floating-point types
  if constexpr (std::is_floating_point<T>::value) {
    if (scientific) {
      out << std::scientific;
    } else {
      out << std::fixed;
    }
    out << std::setprecision(precision);
  }

  for (std::size_t i = 0; i < data.size(); ++i) {
    out << data[i];
    if (i != data.size() - 1) {
      out << (column_vec ? '\n' : delim);
    }
  }
  out << '\n';
  out.close();
}

/// @brief Read a vector of values from a text file.
///
/// @tparam T            The data type of the elements (e.g., int, double, std::string).
///
/// @param file_name     The input file path.
/// @param column_vec    If true expects one element per line; otherwise, all elements on one line, delimited (default: true).
/// @param delim         The delimiter used between elements in row-mode (default: tab).
///
/// @return              A std::vector<T> containing the parsed elements.
///
/// @throws std::runtime_error if the file cannot be opened.
///
/// @details Supports parsing both column-formatted (one value per line) and row-formatted (delimited) input files.
///          Skips lines or tokens that cannot be parsed into type T.
template<typename T>
std::vector<T> read_vector(const std::string& file_name,
                           const bool column_vec = true,
                           const char delim = '\t') {
  std::ifstream in(file_name);
  if (!in.is_open()) {
    throw std::runtime_error("file_io::read_vector - unable to open file " + file_name);
  }

  std::vector<T> result;
  std::string line;

  if (column_vec) {
    // One element per line
    while (std::getline(in, line)) {
      std::istringstream iss(line);
      T value;
      if (iss >> value) {
        result.push_back(value);
      }
    }
  } else {
    // All elements on a single line, separated by delim
    if (std::getline(in, line)) {
      std::istringstream iss(line);
      std::string token;
      while (std::getline(iss, token, delim)) {
        std::istringstream converter(token);
        T value;
        if (converter >> value) {
          result.push_back(value);
        }
      }
    }
  }

  in.close();
  return result;
}

} // namespase file_io
