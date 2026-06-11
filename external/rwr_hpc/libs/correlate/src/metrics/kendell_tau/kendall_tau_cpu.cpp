#include "kendall_tau_cpu.hpp"
#include <stdexcept>
#include <cmath>
#include <algorithm>

#ifdef USE_OPENMP
#include <omp.h>
#endif

namespace correlate::kendall_tau {

// Struct to hold pair counting results
struct TauCounts {
  long long concordant = 0;
  long long discordant = 0;
  long long ties_x = 0;
  long long ties_y = 0;
  long long ties_xy = 0;
};

// Merge sort based inversion count (discordant pairs)
long long merge_count(double* y, int left, int right, double* temp) {
  if (left >= right) return 0;
  int mid = (left + right) / 2;

  long long inv = merge_count(y, left, mid, temp)
                + merge_count(y, mid + 1, right, temp);

  int i = left, j = mid + 1, k = 0;

  while (i <= mid && j <= right) {
    if (y[i] <= y[j]) {
      temp[k++] = y[i++];
    } else {
      temp[k++] = y[j++];
      inv += (mid - i + 1); // discordant pairs
    }
  }
  while (i <= mid) { temp[k++] = y[i++]; }
  while (j <= right) {temp[k++] = y[j++]; }
  for (i = 0; i < k; i++) { y[left + i] = temp[i]; }
  return inv;
}

// Count concordant, discordant, and ties
TauCounts count_concordant_discordant(const double* x, const double* y, const std::size_t M, double* temp_y) {
  std::vector<std::size_t> idx(M);
  for (std::size_t i = 0; i < M; ++i) {
    idx[i] = i;
  }

  // Sort indices by x; break ties with y
  auto sorter_break_ties_with_y = [&] (std::size_t i, std::size_t j) {
    return (x[i] < x[j]) || ((x[i] == x[j]) && (y[i] < y[j]));
  };
  std::sort(idx.begin(), idx.end(), sorter_break_ties_with_y);

  // Prepare y sorted according to x
  for (std::size_t i = 0; i < M; ++i) {
    temp_y[i] = y[idx[i]];
  }

  // discprdant pairs
  long long D = merge_count(temp_y, 0, M-1, temp_y);

  // Total possivle pairs
  long long total_pairs = (long long)M * (M - 1) / 2;

  // Count ties in X
  long long Tx = 0;
  for (std::size_t i = 0; i < M;) {
    std::size_t j = i + 1;
    while (j < M && x[idx[i]] == x[idx[j]]) {
      ++j;
    }
    long long t = j - 1;
    Tx += t * (t-1) / 2;
    i = j;
  }
  
  // Count ties in Y
  
}


// Kendall's tau-b for two vectors using pointers
double kendall_tau_b(const double* x, const double *y, const std::size_t M, double* temp_y) {
  TauCounts counts = count_concordant_discordant(x, y, M, temp_y);

  long long numerator = counts.concordant - counts.discordant;
  long long denominator = std::sqrt( (counts.concordant + counts.discordant + counts.ties_x) *
                                     (counts.concordant + counts.discordant + counts.ties_x));
  return denominator == 0 ? 0.0 : static_cast<double>(numerator) / denominator;
}

void local_cpu(
  std::vector<double>& out,
  std::vector<double>& data,
  const std::size_t M,
  const std::size_t N,
  const bool inplace
) {
  if (data.size() != M * N) {
    throw std::invalid_argument("kendall_tau::local_cpu - data size does not equal M * N");
  }
  if (out.size() != N * N) {
    throw std::invalid_argument("kendall_tau::local_cpu - out size does not equal N * N");
  }

  std::vector<double> temp_y(M); // reusable buffer

  #ifdef USE_OPENMP
  #pragma omp parallel for schedule(dynamic)
  #endif
  for (std::size_t i = 0; i < N; ++i) {
    const double* xi = data.data() + i * M;

    for (std::size_t j = i; j < N; ++j) {
      const double* yi = data.data() + j * M;
      double tau = kendall_tau_b(xi, yi, M, temp_y.data());
      out[i * N + j] = tau;
      out[j * N + i] = tau;
    }
  }
}

void local_distance_cpu(
  std::vector<double>& out,
  std::vector<double>& data,
  const std::size_t M,
  const std::size_t N,
  const bool inplace
) {
  if (data.size() != M * N) {
    throw std::invalid_argument("kendall_tau::local_distance_cpu - data size does not equal M * N");
  }
  if (out.size() != N * N) {
    throw std::invalid_argument("kendall_tau::local_distance_cpu - out size does not equal N * N");
  }

  for (auto& o : out) { o = 1.0; }


}

} // namespace correlate::kendall_tau
