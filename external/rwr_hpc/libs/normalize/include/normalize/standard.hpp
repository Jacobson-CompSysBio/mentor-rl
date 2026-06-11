/// @file standardize_scaler.hpp
/// @brief Column-wise z-score standardization (mean 0, std 1) for dense column-major matrices.
///
/// Provides CPU and (optionally) GPU-accelerated methods
///
/// @author Ken Smith
/// @date 2025-09-24 

#pragma once

#include <vector>

/// @namespace normalize
/// @brief Namespace for column-wise standardization routines for dense data.
namespace normalize::standard {

/// @brief Perform z-score standardization (mean 0, std 1) across each column.
/// 
/// This function dispatches to either a CPU or GPU implementation based on runtime availability
/// and user preference. It operates in-place on a dense matrix flattened in column-major order.
///
/// @param data    The input/output data to standardize (size M × N, column-major).
/// @param M       Number of rows in the matrix.
/// @param N       Number of columns in the matrix.
/// @param use_gpu If true and GPU support is available, runs on GPU; otherwise falls back to CPU (default: true).
///
/// @note This function standardizes each column independently using Welford’s method.
void fit_transform(std::vector<double>& data, const std::size_t M, const std::size_t N, bool use_gpu = true);

/// @cond USE_HIP
// #ifdef USE_HIP
/// @brief Standardize a matrix already resident on the GPU device (in-place).
///
/// This version of standardize performs column-wise z-score normalization directly on
/// a `double*` buffer in GPU device memory, avoiding any host transfers.
///
/// @param d_data  Pointer to the device memory storing M × N doubles in column-major order.
/// @param M       Number of rows.
/// @param N       Number of columns.
///
/// @throws std::runtime_error if compiled without HIP support.
///
/// @note This function standardizes each column independently using Welford’s method.
void fit_transform_device(double* d_data, const std::size_t M, const std::size_t N);
// #endif
/// @endcond

} // namespace normalize
