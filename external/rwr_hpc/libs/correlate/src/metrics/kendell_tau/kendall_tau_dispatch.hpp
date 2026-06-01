#include <correlate/kendall_tau.hpp>

#include <stdexcept>

#include "kendall_tau_seams.hpp"
#include "kendall_tau_cpu.hpp"

#ifdef USE_HIP
#include "../../common/hip_common.hpp"
#include "../../common/hip_seams.hpp"
#include "kendall_tau_gpu.hpp"
#endif

namespace correlate::kendall_tau::seams {

LocalCpuFn local_cpu_fn = [](std::vector<double>& out,
                             std::vector<double>& data,
                             std::size_t M, std::size_t N,
                             bool inplace)
{
  kendall_tau::local_cpu(out, data, M, N, inplace);
};

LocalDistanceCpuFn local_distance_cpu_fn = [](std::vector<double>& out,
                                              std::vector<double>& data,
                                              std::size_t M, std::size_t N,
                                              bool inplace)
{
  kendall_tau::local_distance_cpu(out, data, M, N, inplace);
};

#ifdef USE_HIP
// Bind GPU seam to the real GPU implementation by default.
LocalGpuFn local_gpu_fn = [](std::vector<double>& out,
                             const std::vector<double>& data,
                             std::size_t M,
                             std::size_t N)
{
  kendall_tau::local_gpu(out, data, M, N);
};

// Bind GPU seam to the real GPU implementation by default.
LocalDistanceGpuFn local_distance_gpu_fn = [](std::vector<double>& out,
                                              const std::vector<double>& data,
                                              std::size_t M,
                                              std::size_t N)
{
  kendall_tau::local_distance_gpu(out, data, M, N);
};
#endif

} // namespace correlate::kendall_tau::seams

namespace correlate::kendall_tau {

void local(
  std::vector<double>& out,
  std::vector<double>& data,
  const std::size_t M,
  const std::size_t N,
  const bool inplace,
  const bool use_gpu
) {
#ifdef USE_HIP
  if (use_gpu && correlate::hip::seams::hip_available_fn()) {
    seams::local_gpu_fn(out, data, M, N);
    return;
  }
#endif
  seams::local_cpu_fn(out, data, M, N, inplace);
}

void local_distance(
  std::vector<double>& out,
  std::vector<double>& data,
  const std::size_t M,
  const std::size_t N,
  const bool inplace,
  const bool use_gpu
) {
#ifdef USE_HIP
  if (use_gpu && correlate::hip::seams::hip_available_fn()) {
    seams::local_distance_gpu_fn(out, data, M, N);
    return;
  }
#endif
  seams::local_distance_cpu_fn(out, data, M, N, inplace);
}

} // namespace correlate::kendall_tau
