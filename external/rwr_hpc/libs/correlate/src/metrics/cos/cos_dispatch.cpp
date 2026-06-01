#include <correlate/cos.hpp>

#include <stdexcept>

#include "cos_seams.hpp"
#include "cos_cpu.hpp"

namespace correlate::cos::seams {

LocalCorColCpuFn local_cor_col_cpu_fn = [](std::vector<double>& out /* Nx1 */,
                                           const std::size_t offset,
                                           const std::vector<double>& X /* MxN */,
                                           const std::vector<double>& Y /* MxN */,
                                           const std::size_t M,
                                           const std::size_t N,
                                           const double alpha,
                                           const double beta)
{
  cos::local_corresponding_columns_cpu(out, offset, X, Y, M, N, alpha, beta);
};

} // namespace correlate::cos::seams

namespace correlate::cos {

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
  seams::local_cor_col_cpu_fn(out, offset, X, Y, M, N, alpha, beta);
}

} // namespace correlate::cos
