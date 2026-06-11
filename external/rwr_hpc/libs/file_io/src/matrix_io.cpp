#include "file_io/matrix_io.hpp"
#include <stdexcept>
#include <fstream>
#include <iomanip>
#include <filesystem>
#include <map>
#include <utility>

#include "file_io/binary_io.hpp"

namespace file_io {

void print_column_major_matrix(
  const std::string& file_name,
  const std::vector<double>& matrix,
  const std::size_t n_rows,
  const std::size_t n_cols,
  const std::vector<std::string>& row_labels,
  const std::vector<std::string>& col_labels,
  const int precision,
  const bool scientific,
  const bool transpose)
{
  // Check that encodings is the expected length
  if (matrix.size() != n_rows * n_cols) {
    throw std::invalid_argument("file_io::print_column_major_matrix - matrix size does not match dimensions: " + std::to_string(matrix.size()) + " vs. " + std::to_string(n_rows * n_cols));
  }
  if (!row_labels.empty() && row_labels.size() != n_rows) {
    throw std::invalid_argument("file_io::print_column_major_matrix - row_lables size does not match number of rows");
  }
  if (!col_labels.empty() && col_labels.size() != n_cols) {
    throw std::invalid_argument("file_io::print_column_major_matrix - col_lables size does not match number of columns");
  }
  
  std::ofstream out(file_name);
  if (!out) {
    throw std::runtime_error("file_io::print_column_major_matrix - could not open file for writing: " + file_name);
  }

  if (scientific) {
    out << std::scientific;
  } else {
    out << std::fixed;
  }
  out << std::setprecision(precision);
  
  if (transpose) {
    if (!row_labels.empty()) {
      if (!col_labels.empty()) {
        out << "INDEX" << '\t';
      }
      // for (const auto& label : row_labels) {
      for (std::size_t row = 0; row < n_rows - 1; ++row) {
        out << row_labels[row] << '\t';
      }
      out << row_labels[n_rows-1] << '\n';
    }

    for (std::size_t col = 0; col < n_cols; ++col) {
      if (!col_labels.empty()) {
        out << col_labels[col] << '\t';
      }
        
      for (std::size_t row = 0; row < n_rows - 1; ++row) {
        // Transpose: (row, col) becomes (col, row)
        out << matrix[col * n_rows + row] << '\t';
      }
      out << matrix[col * n_rows + (n_rows-1)] << '\n';
    }
  } else {
    if (!col_labels.empty()) {
      if (!row_labels.empty()) {
        out << "INDEX" << '\t';
      }
      // for (const auto& label : col_labels) {
      for (std::size_t col = 0; col < n_cols - 1; ++col) {
        out << col_labels[col] << '\t';
      }
      out << col_labels[n_cols-1] << '\n';
    }
    
    for (std::size_t row = 0; row < n_rows; ++row) {
      if (!row_labels.empty()) {
        out << row_labels[row] << '\t';
      }

      for (std::size_t col = 0; col < n_cols - 1; ++col) {
        out << matrix[col * n_rows + row] << '\t';
      }
      out << matrix[(n_cols-1) * n_rows + row] << '\n';
    }
  }

  out.close();
}


void print_column_major_matrix(
  const std::string& file_name,
  const std::vector<bool>& matrix,
  const std::size_t n_rows,
  const std::size_t n_cols,
  const std::vector<std::string>& row_labels,
  const std::vector<std::string>& col_labels,
  const bool transpose)
{
  // Check that encodings is the expected length
  if (matrix.size() != n_rows * n_cols) {
    throw std::invalid_argument("file_io::print_column_major_matrix - matrix size does not match dimensions: " + std::to_string(matrix.size()) + " vs. " + std::to_string(n_rows * n_cols));
  }
  if (!row_labels.empty() && row_labels.size() != n_rows) {
    throw std::invalid_argument("file_io::print_column_major_matrix - row_lables size does not match number of rows");
  }
  if (!col_labels.empty() && col_labels.size() != n_cols) {
    throw std::invalid_argument("file_io::print_column_major_matrix - col_lables size does not match number of columns");
  }

  std::ofstream out(file_name);
  if (!out) {
    throw std::runtime_error("file_io::print_column_major_matrix - could not open file for writing: " + file_name);
  }
  
  if (transpose) {
    if (!row_labels.empty()) {
      if (!col_labels.empty()) {
        out << "INDEX" << '\t';
      }
      // for (const auto& label : row_labels) {
      for (std::size_t row = 0; row < n_rows - 1; ++row) {
        out << row_labels[row] << '\t';
      }
      out << row_labels[n_rows-1] << '\n';
    }

    for (std::size_t col = 0; col < n_cols; ++col) {
      if (!col_labels.empty()) {
        out << col_labels[col] << '\t';
      }
        
      for (std::size_t row = 0; row < n_rows - 1; ++row) {
        // Transpose: (row, col) becomes (col, row)
        out << matrix[col * n_rows + row] << '\t';
      }
      out << matrix[col * n_rows + (n_rows-1)] << '\n';
    }
  } else {
    if (!col_labels.empty()) {
      if (!row_labels.empty()) {
        out << "INDEX" << '\t';
      }
      // for (const auto& label : col_labels) {
      for (std::size_t col = 0; col < n_cols - 1; ++col) {
        out << col_labels[col] << '\t';
      }
      out << col_labels[n_cols-1] << '\n';
    }
    
    for (std::size_t row = 0; row < n_rows; ++row) {
      if (!row_labels.empty()) {
        out << row_labels[row] << '\t';
      }

      for (std::size_t col = 0; col < n_cols - 1; ++col) {
        out << matrix[col * n_rows + row] << '\t';
      }
      out << matrix[(n_cols-1) * n_rows + row] << '\n';
    }
  }

  out.close();
}

void print_column_major_distance_matrix(const std::string& file_name,
                                        const std::vector<double>& matrix,
                                        const std::size_t dim,
                                        const std::vector<std::string>& labels,
                                        const int precision,
                                        const bool scientific,
                                        const DIST_MATRIX_MODE mode) {
 // Check that encodings is the expected length
  if (matrix.size() != dim * dim) {
    throw std::invalid_argument("file_io::print_column_major_distance_matrix - matrix size does not match dimensions");
  }
  if (!labels.empty() && labels.size() != dim) {
    throw std::invalid_argument("file_io::print_column_major_distance_matrix - lables size does not match dimension");
  }

  std::ofstream out(file_name);
  if (!out) {
    throw std::runtime_error("file_io::print_column_major_distance_matrix - could not open file for writing: " + file_name);
  }

  if (scientific) {
    out << std::scientific;
  } else {
    out << std::fixed;
  }

  out << std::setprecision(precision);
  
  if (!labels.empty()) {
    out << "INDEX" << '\t';
    for (std::size_t col = 0; col < dim - 1; ++col) {
      out << labels[col] << '\t';
    }
    out << labels[dim-1] << '\n';
  }
  
  if (mode == DIST_MATRIX_MODE::ALL) {
    for (std::size_t row = 0; row < dim; ++row) {
      if (!labels.empty()) {
        out << labels[row] << '\t';
      }

      for (std::size_t col = 0; col < dim - 1; ++col) {
        out << matrix[col * dim + row] << '\t';
      }
      out << matrix[(dim-1) * dim + row] << '\n';
    }
  } else if (mode == DIST_MATRIX_MODE::UPPER) {
    for (std::size_t row = 0; row < dim; ++row) {
      if (!labels.empty()) {
        out << labels[row] << '\t';
      }

      for (std::size_t col = 0; col < dim - 1; ++col) {
        if (col > row) {
          out << matrix[col * dim + row] << '\t';
        } else {
          out << "NA" << "\t";
        }
      }
      if ((dim-1) > row) {
        out << matrix[(dim-1) * dim + row] << '\n';
      } else {
        out << "NA" << '\n';
      }
    }
  } else if (mode == DIST_MATRIX_MODE::LOWER) {
    for (std::size_t row = 0; row < dim; ++row) {
      if (!labels.empty()) {
        out << labels[row] << '\t';
      }

      for (std::size_t col = 0; col < dim - 1; ++col) {
        if (col < row) {
          out << matrix[col * dim + row] << '\t';
        } else {
          out << "NA" << "\t";
        }
      }
      out << "NA" << '\n';
    }
  }

  out.close();
}

// void print_lower_triangle_from_upper(const std::string& file_name,
//                            const std::vector<double>& upper_tri,
//                            const std::size_t dim,
//                            const std::vector<std::string>& row_labels,
//                            const std::vector<std::string>& col_labels,
//                            const bool print_diagonal,
//                            const std::string& na_token,
//                            const int precision) {
//   // Check that labels are either empty or match provided dimensions
//   if (!row_labels.empty() && row_labels.size() != dim) {
//     fprintf(stderr, "n_row_lables: %lu vs. %lu\n", row_labels.size(), dim);
//     throw std::invalid_argument("file_io::print_lower_triangle_from_upper - row_lables size does not match number of rows");
//   }
//   if (!col_labels.empty() && col_labels.size() != dim) {
//     fprintf(stderr, "n_col_lables: %lu vs. %lu\n", col_labels.size(), dim);
//     throw std::invalid_argument("file_io::print_lower_triangle_from_upper - col_lables size does not match number of columns");
//   }
//   // Check matrix size
//   if (upper_tri.size() != (dim * dim + dim) / 2) {
//     throw std::invalid_argument("file_io::print_lower_triangle_from_upper - matrix size does not match expected dimensions");
//   }

//   // Open file
//   std::ofstream out(file_name);
//   if (!out) {
//     throw std::runtime_error("file_io::print_lower_triangle_matrix - could not open file for writing: " + file_name);
//   }

//   auto get_index = [dim](std::size_t i, std::size_t j) -> std::size_t {
//     return i * dim - (i * (i + 1)) / 2 + j;  // maps (i, j) where i <= j
//   };

//   if (!col_labels.empty()) {
//     if (!row_labels.empty()) {
//       out << "INDEX" << '\t';
//     }
//     for (std::size_t col = 0; col < dim - 1; ++col) {
//       out << col_labels[col] << '\t';
//     }
//     out << col_labels[dim-1] << '\n';
//   }
  
//   std::size_t idx = 0;
//   for (std::size_t i = 0; i < dim; ++i) {
//     if (!row_labels.empty()) {
//       out << row_labels[i] << '\t';
//     }

//     for (std::size_t j = 0; j < dim - 1; ++j) {
//       if (j < i || (print_diagonal && j == i)) {
//         std::size_t idx = (i <= j) ? get_index(i, j) : get_index(j, i);
//         out << std::fixed << std::setprecision(precision)
//                           << upper_tri[idx] << '\t';
//       } else {
//         out << na_token << '\t';
//       }
//     }
//     out << '\n';
//   }

//   out.close();
// }

void read_matrix_from_binary_blocks(std::vector<double>& output_matrix,
                                    const std::string& directory,
                                    const std::string& file_prefix,
                                    const std::vector<std::size_t>& N_ranks) {
  // Create map for offset based on block number 
  std::map<int, size_t> block_offsets;
  size_t offset = 0;
  for (size_t i = 0; i < N_ranks.size(); ++i) {
    block_offsets[i] = offset;
    offset += N_ranks[i];
  }

  // Allocate output matrix
  const size_t N = offset;
  output_matrix.resize(N * N, std::numeric_limits<double>::quiet_NaN());

  // Find all block files in 'directory' of the form <file_prefix>_#_#.bin. Return all <#, #> pairs
  auto block_file_nums = find_block_files(directory, file_prefix);

  // Loop through each block_numbers and read them in
  for (auto [i, j] : block_file_nums) {
    size_t Ni = N_ranks[i];
    size_t Nj = N_ranks[j];
    size_t row_base = block_offsets[i];
    size_t col_base = block_offsets[j];

    std::string file_name = file_prefix + "_" + std::to_string(i) + "_" + std::to_string(j) + ".bin";
    std::filesystem::path full_path = directory / std::filesystem::path(file_name);

    auto block = load_binary_block<double>(full_path.string(), Ni, Nj);

    // Copy block to output_matrix in column-major format
    for (size_t c = 0; c < Nj; ++c) {
      for (size_t r = 0; r < Ni; ++r) {
        size_t global_r = row_base + r;
        size_t global_c = col_base + c;
        output_matrix[global_c * N + global_r] = block[c * Ni + r];
      }
    }
  }
}

void write_column_major_matrix_with_yaml(
  const std::string& bin_file,
  const std::string& yaml_file,
  const std::vector<double>& matrix,
  std::size_t n_rows,
  std::size_t n_cols,
  const std::vector<std::string>& row_labels,
  const std::vector<std::string>& col_labels,
  int precision,
  bool scientific,
  bool transpose
) {
  // --- 1. Check sizes ---
  if (matrix.size() != n_rows * n_cols) {
    throw std::invalid_argument("file_io::write_column_major_matrix_with_yaml - matrix size does not match dimensions: " + std::to_string(matrix.size()) + " vs. " + std::to_string(n_rows * n_cols));
  }
  if (!row_labels.empty() && row_labels.size() != n_rows) {
    throw std::invalid_argument("file_io::write_column_major_matrix_with_yaml - row_lables size does not match number of rows");
  }
  if (!col_labels.empty() && col_labels.size() != n_cols) {
    throw std::invalid_argument("file_io::write_column_major_matrix_with_yaml - col_lables size does not match number of columns");
  }

  // --- 2. Write binary file ---
  std::ofstream out_bin(bin_file, std::ios::binary);
  if (!out_bin) throw std::runtime_error("file_io::write_column_major_matrix_with_yaml - Cannot open binary file for writing");

  // Write raw column-major doubles
  out_bin.write(reinterpret_cast<const char*>(matrix.data()),
                matrix.size() * sizeof(double));
  out_bin.close();

  // --- 3. Write YAML metadata ---
  std::ofstream out_yaml(yaml_file);
  if (!out_yaml) throw std::runtime_error("Cannot open YAML file for writing");

  out_yaml << "n_rows: " << n_rows << "\n";
  out_yaml << "n_cols: " << n_cols << "\n";
  out_yaml << "precision: " << precision << "\n";
  out_yaml << "scientific: " << (scientific ? "true" : "false") << "\n";
  out_yaml << "transpose: " << (transpose ? "true" : "false") << "\n";

  // Optional row labels
  if (!row_labels.empty()) {
    out_yaml << "row_labels:\n";
    for (const auto& label : row_labels) {
      out_yaml << "  - \"" << label << "\"\n";
    }
  }

  // Optional column labels
  if (!col_labels.empty()) {
    out_yaml << "col_labels:\n";
    for (const auto& label : col_labels) {
      out_yaml << "  - \"" << label << "\"\n";
    }
  }

  out_yaml.close();
}

} // namespace file_io
