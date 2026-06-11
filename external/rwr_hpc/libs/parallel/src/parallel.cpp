#include "parallel/parallel.hpp"
#include <limits>
#include "parallel_internal.hpp"
#include <utils/vector_utils.hpp>

namespace parallel {

int get_comm_rank(MPI_Comm comm) {
  int world_rank;
  MPI_Comm_rank(comm, &world_rank);
  return world_rank;
}

int get_comm_size(MPI_Comm comm) {
  int world_size;
  MPI_Comm_size(comm, &world_size);
  return world_size;
}

std::vector<std::size_t> gather_n_elements(
  const std::size_t n_local_elements,
  const int world_size,
  const int root,
  MPI_Comm comm)
{
  std::vector<std::size_t> n_elements;
  if (get_comm_rank(comm) == root) {
    n_elements.resize(world_size);
  }

  MPI_Gather(
    &n_local_elements, 1, MPI_UNSIGNED_LONG_LONG,
    n_elements.data(), 1, MPI_UNSIGNED_LONG_LONG,
    root, comm
  );

  return n_elements;
}

void bcast_n_elements(std::vector<std::size_t>& n_elements,
                      const int root,
                      MPI_Comm comm) {
  int comm_size = get_comm_size(comm);
  int comm_rank = get_comm_rank(comm);
  
  if (comm_rank != root) {
    n_elements.resize(comm_size);
  }

  MPI_Bcast(n_elements.data(), comm_size, MPI_UNSIGNED_LONG_LONG, root, comm);
}

void gather(
  std::vector<double>& output,
  const std::vector<double>& local_vector,
  const int root,
  MPI_Comm comm)
{
  gather(output, local_vector.data(), local_vector.size(), root, comm);
}

void gather(
  std::vector<double>& output,
  const double* local_vector,
  const std::size_t local_size,
  const int root,
  MPI_Comm comm)
{
  int comm_size = get_comm_size(comm);

  // Gather local number of elements
  std::vector<std::size_t> counts = gather_n_elements(local_size, comm_size, root, comm);
  bcast_n_elements(counts, root, comm);
  
  bool need_to_batch_transfer = false, can_use_gather = true;
  std::size_t n_total_elements = 0;
  for (auto c : counts) {
    if (c > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
      need_to_batch_transfer = true;
    }
    n_total_elements += c;
  }

  if (n_total_elements > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    can_use_gather = false;
  }

  if (can_use_gather) {
    gather_with_gatherv(output, local_vector, local_size, n_total_elements, counts, root, comm);
  } else if (need_to_batch_transfer) {
    gather_with_recv_batched(output, local_vector, local_size, n_total_elements, counts, root, comm);
  } else {
    gather_with_recv(output, local_vector, local_size, n_total_elements, counts, root, comm);
  }
}

void gather_and_row_join_column_major(
  std::vector<double>& output,
  const std::vector<double>& local_vector,
  const std::size_t n_cols,
  const int root,
  MPI_Comm comm)
{
  int comm_rank = get_comm_rank(comm);
  int comm_size = get_comm_size(comm);
  int local_rows = static_cast<int>(local_vector.size() / n_cols);
  
  std::vector<int> row_counts(comm_size);
  MPI_Gather(&local_rows, 1, MPI_INT,
             row_counts.data(), 1, MPI_INT,
             0, comm);

  std::vector<int> displs(comm_size);
  std::vector<int> recv_counts(comm_size);
  std::size_t total_rows = 0;

  if (comm_rank == 0) {
    for (int i = 0; i < comm_size; ++i) {
      recv_counts[i] = row_counts[i];
      displs[i] = total_rows;
      total_rows += row_counts[i];
    }
    output.resize(total_rows * n_cols); // column-major result
  }

  for (std::size_t j = 0; j < n_cols; ++j) {
    const double* local_col = &local_vector[j * local_rows];  // column-major

    // Gather column j to position starting at j * total_rows
    MPI_Gatherv(local_col, local_rows, MPI_DOUBLE,
                (comm_rank == 0) ? &output[j * total_rows] : nullptr,
                recv_counts.data(), displs.data(), MPI_DOUBLE,
                0, comm);
  }
}

void gather_with_gatherv(
  std::vector<double>& output,
  const double* local_vector,
  const std::size_t local_size,
  const std::size_t n_total_elements,
  const std::vector<std::size_t>& counts,
  const int root,
  MPI_Comm comm)
{
  int comm_rank = get_comm_rank(comm);
  int comm_size = get_comm_size(comm);

  std::vector<int> displs_int, counts_int;

  if (comm_rank == root) {
    // Cast counts vector from std::size_t to int
    counts_int = utils::safe_cast_vector<int>(counts);

    // Gather all elements using MPI_Gatherv
    displs_int = calc_displacement(counts_int);
    output.resize(n_total_elements);
  };
  
  MPI_Gatherv(
    local_vector, static_cast<int>(local_size), MPI_DOUBLE,
    (comm_rank == root) ? output.data() : nullptr,
    counts_int.data(), displs_int.data(), MPI_DOUBLE,
    root, comm
  );
}

void gather_with_recv(
  std::vector<double>& output,
  const double* local_vector,
  const std::size_t local_size,
  const std::size_t n_total_elements,
  const std::vector<std::size_t>& counts,
  const int root,
  MPI_Comm comm)
{
  int comm_rank = get_comm_rank(comm);
  int comm_size = get_comm_size(comm);

  if (comm_rank == root) {
    output.resize(n_total_elements);
    
    // Gether all elements using a single MPI_Send/MPI_Recv per MPI rank
    auto displs = calc_displacement(counts);
    auto counts_int = utils::safe_cast_vector<int>(counts);

    std::copy(local_vector, local_vector + local_size, output.begin() + displs[comm_rank]);

    for (int r = 0; r < comm_size; ++r) {
      if (r != root) {
        MPI_Recv(output.data() + displs[r], counts_int[r], MPI_DOUBLE, r, 0, comm, MPI_STATUS_IGNORE);
      }
    }
    
  } else {
    MPI_Send(local_vector, local_size, MPI_DOUBLE, root, 0, comm);
  }
}

void gather_with_recv_batched(
  std::vector<double>& output,
  const double* local_vector,
  const std::size_t local_size,
  const std::size_t n_total_elements,
  const std::vector<std::size_t>& counts,
  const int root,
  MPI_Comm comm,
  std::size_t batch_limit)
{
  int comm_rank = get_comm_rank(comm);
  int comm_size = get_comm_size(comm);

  if (comm_rank == root) {
    auto displs = calc_displacement(counts);
    output.resize(n_total_elements);

    for (int r = 0; r < comm_size; ++r) {
      if (r == root) {
        std::copy(local_vector, local_vector + local_size, output.begin() + displs[r]);
      } else {
        std::size_t remaining = counts[r];
        std::size_t index = 0;
        while (remaining > 0) {
          std::size_t current_batch = std::min(batch_limit, remaining);
          MPI_Recv(output.data() + displs[r] + index, current_batch, MPI_DOUBLE, r, 101, comm, MPI_STATUS_IGNORE);
          index += current_batch;
          remaining -= current_batch;
        }
      }
    }
  } else {
    std::size_t remaining = local_size;
    std::size_t index = 0;
    while (remaining > 0) {
      std::size_t current_batch = std::min(batch_limit, remaining);
      MPI_Send(local_vector + index, current_batch, MPI_DOUBLE, root, 101, comm);
      index += current_batch;
      remaining -= current_batch;
    }
  }
}

void reduce_or_vector_char(
  const std::vector<char>& local,
  std::vector<char>& global,
  MPI_Comm comm
) {
  global.resize(local.size(), 0);

  MPI_Allreduce(
    local.data(),       // send buffer
    global.data(),      // receive buffer
    local.size(),       // number of elements
    MPI_CHAR,                // datatype
    MPI_BOR,                 // bitwise OR
    comm
  );
}


void reduce_sum_vector_vector(
  const std::vector<double>& local,
  std::vector<double>& global,
  MPI_Comm comm 
) {
  global.resize(local.size(), 0);

  MPI_Allreduce(
    local.data(),       // send buffer
    global.data(),      // receive buffer
    local.size(),       // number of elements
    MPI_DOUBLE,         // datatype
    MPI_SUM,            // bitwise OR
    comm
  );
}

} // namespace parallel
