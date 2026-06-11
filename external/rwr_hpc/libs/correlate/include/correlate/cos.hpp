/// @file cos.hpp
/// @brief 
///
/// @author Ken Smith
/// @date 2025-11-20

#pragma once

#include <vector>

/// @namespace correlate
/// @brief Namespace for all correlation and distance methods
namespace correlate {

  /// @namespace correlate::cos
  /// @brief Functions related to cosine similarity, cosine distance, etc.
  namespace cos {
    /// @brief Dispatch function to compute a cosine measure for corresponding columns of two matrices.
    ///
    /// This function computes the cosine measure between matching columns of
    /// two column-major matrices `X` and `Y`, each of size `M × N`.
    /// The result for each column `i` is written to `out[offset + i]` as:
    ///
    ///   out[i] = alpha * cos(X_col_i, Y_col_i) + beta * out[i]
    ///
    /// where `cos` is the cosine similarity of column `i` of `X` and `Y`.
    ///
    /// If `USE_OPENMP` is defined, the column loop is parallelized.
    ///
    /// @param[out] out
    ///     Output buffer with at least `offset + N` elements.
    /// @param[in] offset
    ///     Index within `out` where the first result will be written.
    /// @param[in] X
    ///     Input matrix of size `M * N` stored in column-major order.
    /// @param[in] Y
    ///     Input matrix of size `M * N` stored in column-major order; must match `X` in size.
    /// @param[in] M
    ///     Number of rows in the input matrices.
    /// @param[in] N
    ///     Number of columns in the input matrices.
    /// @param[in] alpha
    ///     Scaling factor applied to the computed cosine similarity.
    /// @param[in] beta
    ///     Scaling factor applied to the existing values in `out`.
    ///
    /// @throws std::invalid_argument
    ///     Thrown if `X` and `Y` have different sizes or if their size does not equal `M * N`.
    /// @throws std::out_of_range
    ///     Thrown if writing at `offset + N` exceeds the size of `out`.
    ///
    /// @note If either column has zero magnitude, the cosine similarity is treated as 0.
    /// @note When OpenMP is enabled, computation over columns is parallelized.
    void local_corresponding_columns(
      std::vector<double>& out,
      const std::size_t offset,
      const std::vector<double>& X,
      const std::vector<double>& Y,
      const std::size_t M,
      const std::size_t N,
      const double alpha = 1.0,
      const double beta = 0.0
    );

  } // namespace cos

} // namespace correlate
