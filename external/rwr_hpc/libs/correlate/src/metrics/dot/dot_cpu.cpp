#include "dot_cpu.hpp"
#include <stdexcept>

#ifdef USE_OPENMP
#include <omp.h>
#endif

namespace correlate::dot {

void local_cpu(
  std::vector<double>& out,
  const std::vector<double>& data,
  const std::size_t M,
  const std::size_t N,
  const double alpha,
  const double beta
) {
  if (data.size() != M * N) {
    throw std::invalid_argument("dot::local_cpu - data size does not equal M * N");
  }
  if (out.size() != N * N) {
    throw std::invalid_argument("dot::local_cpu - out size does not equal N * N");
  }

  auto dot_product = [&](std::size_t i, std::size_t j) {
    double dot = 0.0;
    for (std::size_t k = 0; k < M; ++k) {
      dot += data[i * M + k] * data[j * M + k];
    }
    return dot;
  };

  #ifdef USE_OPENMP
  #pragma omp parallel for
  #endif
  for (std::size_t i = 0; i < N; ++i) {
    // Calculate value on the main diagonal
    double dot = dot_product(i, i);
    out[i * N + i] = alpha * dot + beta * out[i * N + i];

    for (std::size_t j = i + 1; j < N; ++j) {
      double dot = dot_product(i, j);
      out[j * N + i] = alpha * dot + beta * out[j * N + i];
      out[i * N + j] = alpha * dot + beta * out[i * N + j];
    }
  }
}

void local_corresponding_columns_cpu(
  std::vector<double>& out,
  const std::size_t offset,
  const std::vector<double>& X,
  const std::vector<double>&Y,
  const std::size_t M,
  const std::size_t N,
  const double alpha,
  const double beta
) {
  if (X.size() != Y.size()) {
    throw std::invalid_argument("dot::local_corresponding_columns_cpu - X and Y are not the same size");
  }
  if (X.size() != M * N) {
    throw std::invalid_argument("dot::local_corresponding_columns_cpu - data size does not equal M * N");
  }
  if (offset + N > out.size()) {
    throw std::out_of_range("dot::local_corresponding_columns_cpu - offset will result in out of range");
  }

  double* __restrict__ o = out.data() + offset;

  #ifdef USE_OPENMP
  #pragma omp parallel for
  #endif
  for (std::size_t i = 0; i < N; ++i) {
    const double* __restrict__ x = &X[i * M];
    const double* __restrict__ y = &Y[i * M];
    double dot = 0.0;

    #ifdef USE_OPENMP
    #pragma omp simd reduction(+:dot)
    #endif
    for (std::size_t k = 0; k < M; ++k) {
      dot += x[k] * y[k];
    }

    o[i] = alpha * dot + beta * o[i];
  }
}

} // namespace correlate::dot
