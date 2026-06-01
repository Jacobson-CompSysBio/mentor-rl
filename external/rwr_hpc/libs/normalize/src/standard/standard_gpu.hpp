/// @file standard_gpu.hpp
/// @brief Internal HIP (GPU) implementation of column-wise standardization.
///
/// These functions are used internally when GPU support is available via HIP.
///
/// @author Ken Smith
/// @date 2024-07-24 

#pragma once
#include <vector>

/// @namespace normalize
/// @brief Namespace for column-wise standardization routines for dense data.
namespace normalize::standard {

/// @brief GPU-accelerated standardization for column-major data in host memory.
///
/// @param h_data Input/output data in host memory (flattened column-major).
/// @param M      Number of rows.
/// @param N      Number of columns.
///
/// @details This function transfers data to the GPU, performs in-place standardization,
///          and copies results back to host. Called internally by `normalize::standardize`.
void fit_transform_gpu(std::vector<double>& h_data, const std::size_t M, const std::size_t N);

/// @brief Perform in-place standardization on device-resident GPU memory.
///
/// @param d_data Pointer to device memory (flattened column-major matrix).
/// @param M      Number of rows.
/// @param N      Number of columns.
///
/// @details Assumes the buffer is already allocated and initialized on device.
void fit_transform_device_hip(double* d_data, std::size_t M, std::size_t N);

} // namespace normalize::standard

