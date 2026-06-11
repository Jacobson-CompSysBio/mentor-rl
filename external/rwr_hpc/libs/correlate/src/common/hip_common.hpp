#pragma once
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CHECK_HIP(func)                                   \
{                                                         \
  hipError_t status = (func);                             \
  if (status != hipSuccess) {                             \
    std::fprintf(stderr, "HIP API failed at %s:%d: %s\n", \
      __FILE__, __LINE__, hipGetErrorString(status));     \
    std::exit(EXIT_FAILURE);                              \
  }                                                       \
}                                                         \

#define CHECK_HIPBLAS(func)                                              \
{                                                                        \
  hipblasStatus_t status = (func);                                       \
  if (status != HIPBLAS_STATUS_SUCCESS) {                                \
    std::fprintf(stderr, "HIPBLAS API failed at %s:%d with error: %d\n", \
            __FILE__, __LINE__, status);                                 \
    std::exit(EXIT_FAILURE);                                             \
  }                                                                      \
}

namespace correlate {

bool hip_available();

void fill(double* data, const std::size_t count, const double value);

}
