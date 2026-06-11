#include <correlate/dot.hpp>

#include <stdexcept>

#include "dot_seams.hpp"
#include "dot_cpu.hpp"

#ifdef USE_HIP
#include "../../common/hip_common.hpp"
#include "../../common/hip_seams.hpp"
#include "dot_gpu.hpp"
#endif

#ifdef USE_MPI
#include <mpi.h>
#endif

namespace correlate::dot::seams {

LocalCpuFn local_cpu_fn = [](std::vector<double>& out,
                             const std::vector<double>& data,
                             std::size_t M, std::size_t N,
                             double alpha, double beta)
{
  dot::local_cpu(out, data, M, N, alpha, beta);
};

LocalCorrColCpuFn local_corr_col_cpu_fn = [](std::vector<double>& out /* Nx1 */,
                                    const std::size_t offset,
                                    const std::vector<double>& X /* MxN */,
                                    const std::vector<double>& Y /* MxN */,
                                    const std::size_t M,
                                    const std::size_t N,
                                    const double alpha,
                                    const double beta)
{
  dot::local_corresponding_columns_cpu(out, offset, X, Y, M, N, alpha, beta);
};

#ifdef USE_HIP
// Bind GPU seam to the real GPU implementation by default.
// Signature includes initial_output to match your GPU API.
LocalGpuFn local_gpu_fn = [](std::vector<double>& out,
                             const std::vector<double>& data,
                             std::size_t M, std::size_t N,
                             double alpha, double beta,
                             double initial_output)
{
  dot::local_gpu(out, data, M, N, alpha, beta, initial_output);
};
#endif


// #ifdef USE_MPI
// // Bind CPU distributed seam to real CPU implmentation by default.
// DistributedCpuFn distributed_cpu_fn = [](const std::vector<double>& data,
//                                          const std::size_t M,
//                                          const std::size_t N,
//                                          const double alpha,
//                                          const double beta,
//                                          MPI_Comm comm);
// #endif

// #if defined(USE_HIP) && defined(USE_MPI)
// // Bind GPU distributed seam to real GPU implmentation by default.
// DistributedGpuFn distributed_gpu_fn = [](const std::vector<double>& data,
//                                          const std::size_t M,
//                                          const std::size_t N,
//                                          const double alpha,
//                                          const double beta,
//                                          MPI_Comm comm);
// #endif

} // namespace correlate::dot::seams

namespace correlate::dot {

void local(
  std::vector<double>& out,
  const std::vector<double>& data,
  const std::size_t M,
  const std::size_t N,
  const double alpha,
  const double beta,
  const bool use_gpu
) {
#ifdef USE_HIP
  if (use_gpu && correlate::hip::seams::hip_available_fn()) {
    seams::local_gpu_fn(out, data, M, N, alpha, beta, 0.0);
    return;
  }
#endif
  seams::local_cpu_fn(out, data, M, N, alpha, beta);
}

// Output is Nx1
void local_corresponding_columns(
  std::vector<double>& out,
  const std::size_t offset,
  const std::vector<double>& X,
  const std::vector<double>& Y,
  const std::size_t M,
  const std::size_t N,
  const double alpha,
  const double beta
) {
  seams::local_corr_col_cpu_fn(out, offset, X, Y, M, N, alpha, beta);
}

// #ifdef USE_MPI
// void distributed(
//   const std::vector<double>& data,
//   const std::size_t M,
//   const std::size_t N,
//   const double alpha,
//   const double beta,
//   MPI_Comm comm,
//   const bool use_gpu
// ) {
// #ifdef USE_HIP
//   if (use_gpu && correlate::hip::seams::hip_available_fn()) {
//     seams::distributed_gpu_fn(data, M, N, alpha, beta, comm);
//     return;
//   }
// #endif
//   seams::distributed_cpu_fn(data, M, N, alpha, beta, comm);
// }
// #endif

} // namespace correlate::dot
