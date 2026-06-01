// src/metrics/kendall_tau/kendall_tau_seams.hpp
#pragma once
#include <vector>
#include <cstddef>

namespace correlate::kendall_tau::seams {
  
using LocalCpuFn = void(*)(std::vector<double>& /*out NxN*/,
                           std::vector<double>& /*data MxN*/,
                           const std::size_t M,
                           const std::size_t N,
                           const bool inplace);

using LocalDistanceCpuFn = void(*)(std::vector<double>& /*out NxN*/,
                                   std::vector<double>& /*data MxN*/,
                                   const std::size_t M,
                                   const std::size_t N,
                                   const bool inplace);

#ifdef USE_HIP
  using LocalGpuFn = void(*)(std::vector<double>& /*out NxN*/,
                             const std::vector<double>& /*data MxN*/,
                             const std::size_t M,
                             const std::size_t N);
  using LocalDistanceGpuFn = void(*)(std::vector<double>& /*out NxN*/,
                                     const std::vector<double>& /*data MxN*/,
                                     const std::size_t M,
                                     const std::size_t N);

  extern LocalCpuFn local_cpu_fn; // Define it once in a .cpp (see kendall_tau_cpu.cpp).
  extern LocalDistanceCpuFn local_distance_cpu_fn; // Define it once in a .cpp (see kendall_tau_cpu.cpp).
  extern LocalGpuFn local_gpu_fn; // Define it once in a .cpp (see pearson_gpu.cpp).
  extern LocalDistanceGpuFn local_distance_gpu_fn; // Define it once in a .cpp (see kendall_tau_gpu.cpp).
#else
  extern LocalCpuFn local_cpu_fn; // Define it once in a .cpp (see kendall_tau_cpu.cpp).
  extern LocalDistanceCpuFn local_distance_cpu_fn; // Define it once in a .cpp (see kendall_tau_cpu.cpp).
#endif

} // namespace correlate::kendall_tau::seams
