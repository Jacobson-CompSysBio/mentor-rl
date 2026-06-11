#pragma once
#include <vector>
#include <cstddef>

namespace correlate::cos::seams {

// Signature for the CPU corresponding column function
using LocalCorColCpuFn = void(*)(std::vector<double>& out /* Nx1 */,
                                 const std::size_t offset,
                                 const std::vector<double>& X /* MxN */,
                                 const std::vector<double>& Y /* MxN */,
                                 const std::size_t M,
                                 const std::size_t N,
                                 const double alpha,
                                 const double beta);

extern LocalCorColCpuFn local_cor_col_cpu_fn; // Define it once in a .cpp (see dot_cpu.cpp).

} // namespace correlate::cos::seams
