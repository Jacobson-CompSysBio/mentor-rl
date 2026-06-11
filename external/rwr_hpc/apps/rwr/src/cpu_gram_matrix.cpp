#include "rwr/cpu_gram_matrix.hpp"
#include <set>
#include <unordered_map>
#include <utility>
#include <hip/hip_runtime.h>
#include <stdexcept>
#include <string>
#include <omp.h>
#include <cmath>

#include "parallel/parallel.hpp"

// CPU-only: gather all vectors to rank 0 and compute Gram matrix
void compute_cpu_gram_matrix(
  const std::vector<double>& local_vectors,
  std::size_t N,
  std::size_t local_K,
  int my_rank,
  int world_size,
  std::vector<double>& G,
  MPI_Comm comm)
{
  // Gather local vector values from all ranks
  std::vector<double> all_vectors;
  parallel::gather(all_vectors, local_vectors, 0, my_rank, world_size, comm);
  // std::vector<int> counts(world_size), displs(world_size);
  // int n_total_elements = 0;

  // // Step 1: gather local number of elements
  // int n_local_elements = static_cast<int>(local_vectors.size());
  // MPI_Gather(&n_local_elements, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, comm);

  // // Calculate the number of elements received from each process and
  // //  the displacement in which to place the incoming data
  // if (my_rank == 0) {
  //   displs[0] = 0;
  //   for (int r = 1; r < world_size; ++r) {
  //     displs[r] = displs[r - 1] + counts[r - 1];
  //   }
  //   n_total_elements = displs[world_size - 1] + counts[world_size - 1];


  //   for (int r = 0; r < world_size; ++r) {
  //     fprintf(stderr, "rank%i -- num elements: %i - displacement: %i\n", r, counts[r], displs[r]);
  //   }
  //   fprintf(stderr, "total elements: %i\n", n_total_elements);
  // }
  
  // std::vector<double> all_vectors;
  // if (my_rank == 0) {
  //   all_vectors.resize(n_total_elements);
  // }

  // // Step 2: gather all vectors to rank 0
  // MPI_Gatherv(
  //   local_vectors.data(), n_local_elements, MPI_DOUBLE,
  //   all_vectors.data(), counts.data(), displs.data(), MPI_DOUBLE,
  //   0, comm);

  // Step 3: compute Gram matrix on rank 0
  int total_K = all_vectors.size() / N;
  if (my_rank == 0) {
    fprintf(stderr, "Gathered all vectors (%d)\n", total_K);

    G.resize((total_K * (total_K + 1)) / 2);
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < total_K; ++i) {
      for (int j = i; j < total_K; ++j) {
        double dot = 0.0;
        for (std::size_t k = 0; k < N; ++k) {
          dot += all_vectors[i * N + k] * all_vectors[j * N + k];
        }
        std::size_t index = i * total_K - (i * (i + 1)) / 2 + j;
        G[index] = dot;
      }
    }
  }
}

void spearman_distance(const std::vector<double>& local_vectors,
    std::size_t N,
    std::size_t local_K,
    int my_rank,
    int world_size,
    std::vector<double>& G,
    MPI_Comm comm) {
  compute_cpu_gram_matrix(local_vectors, N, local_K, my_rank, world_size, G, comm);

  const std::size_t scale = N - 1;
  #pragma omp parallel for
  for (std::size_t i = 0; i < G.size(); ++i) {
    G[i] = 1.0 - (G[i] / scale);
  }
}