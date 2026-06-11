/// @file kendall_tau.hpp
/// @brief
///
/// @author Ken Smith
/// @date 2025-12-31

#pragma once

#include <vector>

/// @namespace correlate
/// @brief Namespace for all correlation and distance methods
namespace correlate {
  
  namespace kendall_tau {
    void local(
      std::vector<double>& out,
      const std::vector<double>& data,
      const std::size_t M,
      const std::size_t N,
      const bool use_gpu = true
    );

    void local_distance(
      std::vector<double>& out,
      const std::vector<double>& data,
      const std::size_t M,
      const std::size_t N,
      const bool use_gpu = true
    );

  } // namespace kendall_tau

} // namespace correlate
