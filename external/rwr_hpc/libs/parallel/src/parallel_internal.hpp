/// @file parallel_internal.hpp
/// @brief Utilities for MPI-based parallel data exchange and process metadata.
/// 
/// Provides wrappers for common MPI communication patterns (gather, broadcast),
/// as well as helper functions to obtain rank and world size.
///
/// @author Ken Smith
/// @date 2025-07-25

#pragma once

#include <vector>
#include <mpi.h>

namespace parallel {

template <typename T>
std::vector<T> calc_displacement(const std::vector<T>& counts) {
  std::vector<T> displ(counts.size(), 0);

  for (std::size_t i = 1; i < displ.size(); ++i) {
    displ[i] = displ[i-1] + counts[i - 1];
  }

  return displ;
}

void gather_with_gatherv(
  std::vector<double>& output,
  const double* local_vector,
  const std::size_t local_size,
  const std::size_t n_total_elements,
  const std::vector<std::size_t>& counts,
  const int root,
  MPI_Comm comm
);

void gather_with_recv(
  std::vector<double>& output,
  const double* local_vector,
  const std::size_t local_size,
  const std::size_t n_total_elements,
  const std::vector<std::size_t>& counts,
  const int root,
  MPI_Comm comm
);

void gather_with_recv_batched(
  std::vector<double>& output,
  const double* local_vector,
  const std::size_t local_size,
  const std::size_t n_total_elements,
  const std::vector<std::size_t>& counts,
  const int root,
  MPI_Comm comm,
  std::size_t batch_limit = static_cast<std::size_t>(1e6)
);

} // namespace parallel
