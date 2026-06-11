/// @file standardize_cpu_internal.hpp
/// @brief Internal CPU implementation of column-wise z-score standardization.
///
/// Not intended for direct use. Called via normalize::fit_transform dispatch function.
///
/// @author Ken Smith
/// @date 2024-07-24 

#pragma once
#include <vector>

/// @namespace normalize
/// @brief Namespace for column-wise standardization routines for dense data.
namespace normalize::standard {

/// @brief CPU implementation of column-wise z-score standardization using Welford's algorithm.
/// 
/// @param data The input/output matrix, flattened in column-major order.
/// @param M    Number of rows.
/// @param N    Number of columns.
///
/// @details This function modifies `data` in-place so that each column has mean 0 and standard deviation 1.
///          It is intended to be called internally by the dispatch function `normalize::standard_scaler::fit_transform`.
void fit_transform_cpu(std::vector<double>& data, std::size_t M, std::size_t N);

} // namespace normalize::standard
