/// @file parallel.hpp
/// @brief Utilities for MPI-based parallel data exchange and process metadata.
/// 
/// Provides wrappers for common MPI communication patterns (gather, broadcast),
/// as well as helper functions to obtain rank and world size.
///
/// @author Ken Smith
/// @date 2025-07-24 

#pragma once

#include <limits.h>
#include <mpi.h>
#include <stdint.h>
#include <vector>

// https://stackoverflow.com/a/40808411
/// @def CUSTOM_SIZE_T
/// @brief Mapps C++ 'std::size_t' to the corresponding MPI datatype.
///
/// This macro defines the appropriate 'MPI_*' type that matches the underlying
/// representation of 'std::size_t' on the current platform.
#if SIZE_MAX == UCHAR_MAX
  #define CUSTOM_SIZE_T MPI_UNSIGNED_CHAR
#elif SIZE_MAX == USHRT_MAX
  #define CUSTOM_SIZE_T MPI_UNSIGNED_SHORT
#elif SIZE_MAX == UINT_MAX
  #define CUSTOM_SIZE_T MPI_UNSIGNED
#elif SIZE_MAX == ULONG_MAX
  #define CUSTOM_SIZE_T MPI_UNSIGNED_LONG
#elif SIZE_MAX == ULLONG_MAX
  #define CUSTOM_SIZE_T MPI_UNSIGNED_LONG_LONG
#else
  #error "Could not find proper substitution for CUSTOM_SIZE_T"
#endif

/// @namespace parallel
/// @brief Namespace containing MPI-based parallel utilities
namespace parallel {

/// @brief Returns the rank of the calling process in the given communicator.
///
/// @param comm MPI communicator (default: MPI_COMM_WORLD)
///
/// @return Rank of the current process
int get_comm_rank(MPI_Comm comm = MPI_COMM_WORLD);

/// @brief Returns the total number of processes in the communicator.
///
/// @param comm MPI communicator (default: MPI_COMM_WORLD)
///
/// @return Total number of processes
int get_comm_size(MPI_Comm comm = MPI_COMM_WORLD);

/// @brief Gathers the number of local elements on each rank into a single vector on the root.
///
/// @param n_local_elements Number of elements local to this rank
/// @param world_size Total number of MPI ranks
/// @param root The rank where the result is gathered (default: 0)
/// @param comm MPI communicator (default: MPI_COMM_WORLD)
///
/// @return A vector of length `world_size` with the number of elements on each rank (only valid on root)
std::vector<std::size_t> gather_n_elements(
  const std::size_t n_local_elements,
  const int world_size,
  const int root = 0,
  MPI_Comm comm = MPI_COMM_WORLD
);

/// @brief Broadcasts a vector of per-rank element counts from the root to all ranks.
///
/// @param n_elements Vector of counts (valid only on root before broadcast)
/// @param root The rank from which data is broadcast (default: 0)
/// @param comm MPI communicator (default: MPI_COMM_WORLD)
void bcast_n_elements(std::vector<std::size_t>& n_elements, const int root = 0, MPI_Comm comm = MPI_COMM_WORLD);


/// @brief Gathers a vector of doubles from all ranks into a single vector on the root.
///
/// @param output [output] The aggregated result (only valid on root)
/// @param local_vector Vector of local values on each rank
/// @param root The rank where the result is gathered (default: 0)
/// @param comm MPI communicator (default: MPI_COMM_WORLD)
void gather(
  std::vector<double>& output,
  const std::vector<double>& local_vector,
  const int root = 0,
  MPI_Comm comm = MPI_COMM_WORLD
);

/// @brief Gathers a raw pointer-based array of doubles from all ranks into a vector on the root.
///
/// @param output [output] The aggregated result (only valid on root)
/// @param local_vector Pointer to local values
/// @param local_size Number of elements in the local vector
/// @param root The rank where the result is gathered (default: 0)
/// @param comm MPI communicator (default: MPI_COMM_WORLD)
void gather(
  std::vector<double>& output,
  const double* local_vector,
  const std::size_t local_size,
  const int root = 0,
  MPI_Comm comm = MPI_COMM_WORLD
);


/// @brief Gathers local column-major matrices from all ranks and row-concatenates them on the root.
///
/// @param output [output] Flattened matrix after row-joining (only valid on root)
/// @param local_vector Local matrix in column-major format (flattened)
/// @param n_cols Number of columns in the local and global matrix
/// @param root The rank where the result is gathered (default: 0)
/// @param comm MPI communicator (default: MPI_COMM_WORLD)
void gather_and_row_join_column_major(
  std::vector<double>& output,
  const std::vector<double>& local_vector,
  const std::size_t n_cols,
  const int root = 0,
  MPI_Comm comm = MPI_COMM_WORLD
);


void reduce_or_vector_char(
  const std::vector<char>& local,
  std::vector<char>& global,
  MPI_Comm comm = MPI_COMM_WORLD
);

void reduce_sum_vector_vector(
  const std::vector<double>& local,
  std::vector<double>& global,
  MPI_Comm comm = MPI_COMM_WORLD
);

} // namespace parallel
