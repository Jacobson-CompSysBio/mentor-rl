#include "cos_cpu.hpp"
#include <stdexcept>
#include <cmath>

#ifdef USE_OPENMP
#include <omp.h>
#endif

namespace correlate::cos {

void local_corresponding_columns_cpu(
  std::vector<double>& out,
  const std::size_t offset,
  const std::vector<double>& X,
  const std::vector<double>& Y,
  const std::size_t M,
  const std::size_t N,
  const double alpha,
  const double beta
) {
  // Validate inputs
  if (X.size() != Y.size()) {
    throw std::invalid_argument("cos::local_corresponding_columns_cpu - X and Y are not the same size");
  }
  if (X.size() != M * N) {
    throw std::invalid_argument("cos::local_corresponding_columns_cpu - data size does not equal M * N");
  }
  if (offset + N > out.size()) {
    throw std::out_of_range("cos::local_corresponding_columns_cpu - offset will result in out of range");
  }

  double* __restrict__ o = out.data() + offset;

  #ifdef USE_OPENMP
  #pragma omp parallel for
  #endif
  for (std::size_t i = 0; i < N; ++i) {
    const double* __restrict__ x = &X[i * M];
    const double* __restrict__ y = &Y[i * M];
    double cos = 0.0, mag_x = 0.0, mag_y = 0.0;

    for (std::size_t k = 0; k < M; ++k) {
      cos += (x[k] * y[k]);
      mag_x += (x[k] * x[k]);
      mag_y += (y[k] * y[k]);
    }

    // Protect against divide by zero
    if (mag_x == 0.0 || mag_y == 0.0) {
      cos = 0.0;
    } else {
      mag_x = std::sqrt(mag_x);
      mag_y = std::sqrt(mag_y);

      cos /= (mag_x * mag_y);
    }

    o[i] = alpha * cos + beta * o[i];
  }
}

} // namespace correlate::cos
