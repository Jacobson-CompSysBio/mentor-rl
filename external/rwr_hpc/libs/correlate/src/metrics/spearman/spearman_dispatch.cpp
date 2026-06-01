#include <correlate/spearman.hpp>

#include <stdexcept>

#include "spearman_seams.hpp"
#include "spearman_cpu.hpp"

#ifdef USE_HIP
#include "../../common/hip_common.hpp"
#include "../../common/hip_seams.hpp"
#include "spearman_gpu.hpp"
#endif

namespace correlate::spearman::seams {

LocalCpuFn local_cpu_fn = [](std::vector<double>& out,
                             const std::vector<double>& data,
                             const std::size_t M,
                             const std::size_t N)
{
  spearman::local_cpu(out, data, M, N);
};

LocalDistanceCpuFn local_distance_cpu_fn = [](std::vector<double>& out,
                                              const std::vector<double>& data,
                                              const std::size_t M,
                                              const std::size_t N)
{
  spearman::local_distance_cpu(out, data, M, N);
};

#ifdef USE_HIP
// Bind GPU seam to the real GPU implementation by default.
LocalGpuFn local_gpu_fn = [](std::vector<double>& out,
                             const std::vector<double>& data,
                             const std::size_t M,
                             const std::size_t N)
{
  spearman::local_gpu(out, data, M, N);
};

// Bind GPU seam to the real GPU implementation by default.
LocalDistanceGpuFn local_distance_gpu_fn = [](std::vector<double>& out,
                                              const std::vector<double>& data,
                                              const std::size_t M,
                                              const std::size_t N)
{
  spearman::local_distance_gpu(out, data, M, N);
};
#endif

} // namespace correlate::spearman::seams

namespace correlate::spearman {

void local(
  std::vector<double>& out,
  const std::vector<double>& data,
  const std::size_t M,
  const std::size_t N,
  const bool use_gpu
) {
#ifdef USE_HIP
  if (use_gpu && correlate::hip::seams::hip_available_fn()) {
    seams::local_gpu_fn(out, data, M, N);
    return;
  }
#endif
  seams::local_cpu_fn(out, data, M, N);
}

void local_distance(
  std::vector<double>& out,
  const std::vector<double>& data,
  const std::size_t M,
  const std::size_t N,
  const bool use_gpu
) {
#ifdef USE_HIP
  if (use_gpu && correlate::hip::seams::hip_available_fn()) {
    seams::local_distance_gpu_fn(out, data, M, N);
    return;
  }
#endif
  seams::local_distance_cpu_fn(out, data, M, N);
}

} // namespace correlate::spearman
