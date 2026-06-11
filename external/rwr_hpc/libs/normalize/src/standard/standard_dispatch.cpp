#include <normalize/standard.hpp>

#include "standard_cpu.hpp"
#include "standard_seams.hpp"

#ifdef USE_HIP
#include "../common/hip_common.hpp"
#include "../common/hip_seams.hpp"
#include "standard_gpu.hpp"
#endif

namespace normalize::standard::seams {

FitTransformCpuFn fit_transform_cpu_fn = [](std::vector<double>& data,
                                            const std::size_t M,
                                            const std::size_t N)
{
  standard::fit_transform_cpu(data, M, N);
};

#ifdef USE_HIP
FitTransformGpuFn fit_transform_gpu_fn = [](std::vector<double>& data,
                                            const std::size_t M,
                                            const std::size_t N)
{
  standard::fit_transform_gpu(data, M, N);
};
#endif

} // namespace normalize::standard::seams

namespace normalize::standard {

void fit_transform(std::vector<double>& data, const std::size_t M, const std::size_t N, const bool use_gpu) {
#ifdef USE_HIP
  if (use_gpu && normalize::hip::seams::hip_available_fn()) {
    seams::fit_transform_gpu_fn(data, M, N);
    return;
  }
#endif

  seams::fit_transform_cpu_fn(data, M, N);
}

#ifdef USE_HIP
void fit_transform_device(double* d_data, const std::size_t M, const std::size_t N) {
  fit_transform_device_hip(d_data, M, N);
}
#endif

} // namespace normalize::standard
