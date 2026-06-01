#pragma once

#include <vector>
#include <mpi.h>

void compute_cpu_gram_matrix(
    const std::vector<double>& local_vectors,
    std::size_t N,
    std::size_t local_K,
    int my_rank,
    int world_size,
    std::vector<double>& G,
    MPI_Comm comm = MPI_COMM_WORLD);

void spearman_distance(const std::vector<double>& local_vectors,
    std::size_t N,
    std::size_t local_K,
    int my_rank,
    int world_size,
    std::vector<double>& G,
    MPI_Comm comm = MPI_COMM_WORLD);