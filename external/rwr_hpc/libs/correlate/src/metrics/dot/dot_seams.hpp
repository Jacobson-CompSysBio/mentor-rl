#pragma once
#include <vector>
#include <cstddef>

#ifdef USE_MPI
#include <mpi.h>
#endif

namespace correlate::dot::seams {

// Signature for the CPU Gram (X^T X) free function
using LocalCpuFn = void(*)(std::vector<double>& /*out NxN*/,
                           const std::vector<double>& /*data MxN*/,
                           const std::size_t M,
                           const std::size_t N,
                           const double alpha,
                           const double beta);

// Signature for the CPU corresponding column function
using LocalCorrColCpuFn = void(*)(std::vector<double>& out /* Nx1 */,
                                  const std::size_t offset,
                                  const std::vector<double>& X /* MxN */,
                                  const std::vector<double>& Y /* MxN */,
                                  const std::size_t M,
                                  const std::size_t N,
                                  const double alpha,
                                  const double beta);

extern LocalCpuFn local_cpu_fn; // Define it once in a .cpp (see dot_cpu.cpp).
extern LocalCorrColCpuFn local_corr_col_cpu_fn; // Define it once in a .cpp (see dot_cpu.cpp).

#ifdef USE_HIP
// Signature for the GPU Gram (X^T X) free function
using LocalGpuFn = void(*)(std::vector<double>& /*out NxN*/,
                            const std::vector<double>& /*data MxN*/,
                            const std::size_t M, const std::size_t N,
                            const double alpha, const double beta,
                            const double initial_value);

extern LocalGpuFn local_gpu_fn; // Define it once in a .cpp (see dot_gpu.cpp).
#endif

#ifdef USE_MPI
// Signature for the distributed CPU Gran function
using DistributedCpuFn = void(*)(const std::vector<double>& data,
                                 const std::size_t M,
                                 const std::size_t N,
                                 const double alpha,
                                 const double beta,
                                 MPI_Comm comm);

extern DistributedCpuFn distributed_cpu_fn; // Define it once in a .cpp (see dot_cpu.cpp).
#endif

#if defined(USE_MPI) && defined(USE_HIP)
// Signature for the distributed CPU Gran function
using DistributedGpuFn = void(*)(const std::vector<double>& data,
                                 const std::size_t M,
                                 const std::size_t N,
                                 const double alpha,
                                 const double beta,
                                 MPI_Comm comm);

extern DistributedGpuFn distributed_gpu_fn; // Define it once in a .hip (see dot_gpu.hip).
#endif

} // namespace correlate::dot::seams
