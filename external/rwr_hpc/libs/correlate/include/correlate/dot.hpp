/// @file dot.hpp
/// @brief
/// @author Ken Smith
/// @date 2025-11-21

#pragma once

#include <vector>

/// @namespace correlate
/// @brief Namespace for all correlation and distance methods
namespace correlate {
  
  /// @namespace correalte::dot
  /// @brief Functions related to dot product
  namespace dot {

    void local(
      std::vector<double>& out,
      const std::vector<double>& data,
      const std::size_t M,
      const std::size_t N,
      const double alpha = 1.0,
      const double beta = 0.0,
      const bool use_gpu = true
    );

    // Output is Nx1
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

  } // namespace correlate::dot
} // namesapce correlate
