// tests/test_cos_local_dispatch.cpp
#include <gtest/gtest.h>

#include <correlate/cos.hpp>
#include "metrics/cos/cos_seams.hpp"

// ---------- RAII seam guard (restores original function pointers) ----------
template <class Fn>
struct SeamGuard {
  Fn& slot;
  Fn  saved;
  explicit SeamGuard(Fn& s) : slot(s), saved(s) {}
  ~SeamGuard() { slot = saved; }
};

// ---- observable state for the seams (no captures needed) ----
namespace {
  bool g_cpu_called = false;
  bool g_gpu_called = false;
  bool g_hip_avail_called = false;
}

// ============ Tests ============
TEST(CosLocalCorColDispatch, CallsCpu) {
    const std::size_t offset = 0;
  const std::vector<double> X = {0.774590906, 0.513763385, 0.357532009, 0.487371218, 0.050744795,
                                 0.300630147, 0.880460832, 0.779610085, 0.730755625, 0.008541588,
                                 1.0, 2.0, 5.0, -3.0, -1.0, 50.0, 4.0, -3.0, -1.0, 1.0,
                                 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.0, 0.0,
                                 3.0, -4.0, -5.0, 2.0, 4.0, -5.0, -1.0, 0.0, 0.0, 0.0,
                                 0.17271539, 0.382416903, 0.793810151, 0.560014815, 0.864605157,
                                 0.323185059, 0.744819809, 0.683147407, 0.697989807,0.059140425,
                                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,};

  const std::vector<double> Y = {0.642031426, 0.185417287, 0.514525743, 0.20851762, 0.653119113,
                                 0.277094311, 0.797918211, 0.318373245, 0.568765613, 0.687726279,
                                 -1.0, -2.0, -5.0, 3.0, 1.0, -50.0, -4.0, 3.0, 1.0, -1.0,
                                 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 0.0, 0.0,
                                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                 0.729349489, 0.616345872, 0.368910377, 0.849647312, 0.744862772,
                                 0.45806787, 0.407767465, 0.450712837, 0.689680978, 0.12058077,
                                 0.729349489, 0.616345872, 0.368910377, 0.849647312, 0.744862772,
                                 0.45806787, 0.407767465, 0.450712837, 0.689680978, 0.12058077};
  const std::size_t M = 10;
  const std::size_t N = 6;
  const double alpha = 1.0;
  const double beta = 0.0;
  std::vector<double> out(N);
  const std::vector<double> expcected_out = {0.786927332, -1.0, 1.0, 0.0, 0.878965409, 0.0};

  correlate::cos::local_corresponding_columns(out, offset, X, Y, M, N, alpha, beta);

  ASSERT_EQ(out.size(), N);
  for (std::size_t i = 0; i < N; ++i) {
    EXPECT_NEAR(out[i], expcected_out[i], 1e-8);
  }
}
