#include "standard_cpu.hpp"
#include <vector>
#include <stdexcept>
#include <cmath>

#ifdef USE_OPENMP
#include <omp.h>
#endif

namespace normalize::standard {

void fit_transform_cpu(std::vector<double>& data, std::size_t M, std::size_t N) {
  if (data.size() != M * N) {
    throw std::invalid_argument("standard::fit_transform_cpu - data size does not equal M * N");
  }

  #ifdef USE_OPENMP
  #pragma omp parallel for schedule(static)
  #endif
  for (std::size_t j = 0; j < N; ++j) {
    double* vec = &data[j * M];
    double mean = 0.0, M2 = 0.0;

    for (std::size_t i = 0; i < M; ++i) {
      double x = vec[i];
      double delta = x - mean;
      mean += delta / (i + 1);
      M2 += delta * (x - mean);
    }
    double stddev = std::sqrt(M2 / (M - 1));
    if (stddev < 1e-8) stddev = 1.0;

    #ifdef USE_OPENMP
    #pragma omp simd
    #endif
    for (std::size_t i = 0; i < M; ++i) {
      vec[i] = (vec[i] - mean) / stddev;
    }
  }
}

} // namespace normalize::standard