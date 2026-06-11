#pragma once
#include <vector>
#include <cstddef>

namespace normalize::standard::seams {

// Signature for the CPU fit_transform free function
using FitTransformCpuFn = void(*)(std::vector<double>& data,
                                  const std::size_t M,
                                  const std::size_t N);

extern FitTransformCpuFn fit_transform_cpu_fn; // Define it once in a .cpp (see standard.cpp)

#ifdef USE_HIP
// Signature for the GPU fit_transform free function
using FitTransformGpuFn = void(*)(std::vector<double>& data,
                                const std::size_t M,
                                const std::size_t N);

extern FitTransformGpuFn fit_transform_gpu_fn; // Define it once in a .cpp (see standard_scaler.cpp)
#endif

} // namespace normalize::standard::seams
