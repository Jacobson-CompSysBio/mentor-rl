// tests/test_dot_local_dispatch.cpp
#include <gtest/gtest.h>

#include <correlate/dot.hpp>
#include "metrics/dot/dot_seams.hpp"

#ifdef USE_HIP
#include "common/hip_seams.hpp"
#endif

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
TEST(DotLocalDispatch, CallsCpuWhenUseGpuFalse) {
  const std::size_t M = 5, N = 3;
  std::vector<double> out(N*N, 0.0), data(M*N, 1.0);

  g_cpu_called = false;
  g_gpu_called = false;
  g_hip_avail_called = false;

  SeamGuard guard_cpu(correlate::dot::seams::local_cpu_fn);
  correlate::dot::seams::local_cpu_fn = +[](std::vector<double>& out, const std::vector<double>&,
                      std::size_t, std::size_t, double, double )
  {
    g_cpu_called = true;
    // ASSERT_EQ(m, M); ASSERT_EQ(n, N);
    // ASSERT_EQ(data.size(), M * N);
    // ASSERT_EQ(a, 1.0); ASSERT_EQ(b, 0.0);
    // sentinel
    out.assign(N * N, 1.23);
  };

#ifdef USE_HIP
  SeamGuard guard_hip_avail(correlate::hip::seams::hip_available_fn);
  correlate::hip::seams::hip_available_fn = +[]()
  {
    g_hip_avail_called = true;
    return false;
  };

  SeamGuard guard_gpu(correlate::dot::seams::local_gpu_fn);
  correlate::dot::seams::local_gpu_fn = +[](std::vector<double>& out, const std::vector<double>&,
                      std::size_t, std::size_t, double, double, double)
  {
    g_gpu_called = true;
    out.assign(N * N, 9.99); // should not be used in this test
  };
#endif

  correlate::dot::local(out, data, M, N, /*alpha=*/1.0, /*beta=*/0.0, /*use_gpu=*/false);

  EXPECT_TRUE(g_cpu_called);
#ifdef USE_HIP
  EXPECT_FALSE(g_gpu_called);
  EXPECT_FALSE(g_hip_avail_called);
#endif
  for (double v : out) EXPECT_DOUBLE_EQ(v, 1.23);
}

TEST(DotLocalDispatch, CpuFallbackWhenGpuRequestedButUnavailableOrDisabled) {
  const std::size_t M = 5, N = 3;
  std::vector<double> out(N*N, 0.0), data(M*N, 1.0);

  g_cpu_called = false;
  g_gpu_called = false;
  g_hip_avail_called = false;

  SeamGuard guard_cpu(correlate::dot::seams::local_cpu_fn);
  correlate::dot::seams::local_cpu_fn = +[](std::vector<double>& out, const std::vector<double>&,
                      std::size_t, std::size_t, double, double )
  {
    g_cpu_called = true;
    // ASSERT_EQ(m, M); ASSERT_EQ(n, N);
    // ASSERT_EQ(data.size(), M * N);
    // ASSERT_EQ(a, 1.0); ASSERT_EQ(b, 0.0);
    // sentinel
    out.assign(N * N, 1.23);
  };

#ifdef USE_HIP
  SeamGuard guard_hip_avail(correlate::hip::seams::hip_available_fn);
  correlate::hip::seams::hip_available_fn = +[]() -> bool {
    g_hip_avail_called = true;
    return false;
  };

  SeamGuard guard_gpu(correlate::dot::seams::local_gpu_fn);
  correlate::dot::seams::local_gpu_fn = +[](std::vector<double>& out, const std::vector<double>&,
                      std::size_t, std::size_t, double, double, double)
  {
    g_gpu_called = true;
    out.assign(N * N, 9.99); // should not be used in this test
  };
#endif

  correlate::dot::local(out, data, M, N, /*alpha=*/1.0, /*beta=*/0.0, /*use_gpu=*/true);

  EXPECT_TRUE(g_cpu_called);
#ifdef USE_HIP
  EXPECT_FALSE(g_gpu_called);
  EXPECT_TRUE(g_hip_avail_called);
#endif
  for (double v : out) EXPECT_DOUBLE_EQ(v, 1.23);
}

TEST(DotLocalDispatch, GpuPathWhenGpuRequestedAndHipAvailable) {
#ifdef USE_HIP
  const std::size_t M = 5, N = 3;
  std::vector<double> out(N*N, 0.0), data(M*N, 1.0);

  g_cpu_called = false;
  g_gpu_called = false;
  g_hip_avail_called = false;

  SeamGuard guard_cpu(correlate::dot::seams::local_cpu_fn);
  correlate::dot::seams::local_cpu_fn = +[](std::vector<double>& out, const std::vector<double>&,
                      std::size_t, std::size_t, double, double )
  {
    g_cpu_called = true;
    // ASSERT_EQ(m, M); ASSERT_EQ(n, N);
    // ASSERT_EQ(data.size(), M * N);
    // ASSERT_EQ(a, 1.0); ASSERT_EQ(b, 0.0);
    // sentinel
    out.assign(N * N, 1.23);
  };

#ifdef USE_HIP
  SeamGuard guard_hip_avail(correlate::hip::seams::hip_available_fn);
  correlate::hip::seams::hip_available_fn = +[]() -> bool {
    g_hip_avail_called = true;
    return true;
  };

  SeamGuard guard_gpu(correlate::dot::seams::local_gpu_fn);
  correlate::dot::seams::local_gpu_fn = +[](std::vector<double>& out, const std::vector<double>&,
                      std::size_t, std::size_t, double, double, double)
  {
    g_gpu_called = true;
    out.assign(N * N, 9.99); // should not be used in this test
  };
#endif

  correlate::dot::local(out, data, M, N, /*alpha=*/1.0, /*beta=*/0.0, /*use_gpu=*/true);

  EXPECT_FALSE(g_cpu_called);
  EXPECT_TRUE(g_gpu_called);
  EXPECT_TRUE(g_hip_avail_called);
  for (double v : out) EXPECT_DOUBLE_EQ(v, 9.99);
#else
  GTEST_SKIP() << "Test skipped: library not built with HIP.";
#endif
}

TEST(DotLocalDispatch, FullCpuTest) {
  const std::size_t M = 25;
  const std::size_t N = 5;
  std::vector<double> data = {-0.79681066,  0.0499037 , -0.45662631, -0.32100215, -0.03251983,
                              -0.59374907,  0.57441701, -0.68141773, -0.72210498,  3.22695578,
                              -0.58850394, -0.67991912, -0.63795805, -0.72232977, -0.25431403,
                              -0.40642289,  2.08051953,  1.39865222,  0.02742455,  0.02742455,
                              -0.73244538,  1.14388861, -0.51057625,  0.3196534 , -0.71213922,
                              0.07217628, -0.91038683, -0.34190389, -0.60859959, -0.05853971,
                              -0.47612903,  1.09860381, -0.81914883, -0.85432809,  0.28272552,
                              -0.82625486, -0.5542077 ,  0.1774509 , -0.83634368,  3.16023176,
                              1.37933613,  0.02831185,  0.04585762, -0.44191478, -0.39190933,
                              -0.78756644,  1.57233959,  1.37056324, -0.5770172 , -0.70334674,
                              2.02360187,  2.64072002, -0.04699192, -0.7209499 , -0.62513419,
                              -0.76966976, -0.86873346, -0.80864564, -1.00937144,  0.29079906,
                              1.29280406,  0.08292769,  0.36712684, -0.89634138,  0.18686338,
                              -0.53743845,  0.27131112, -0.4887186 ,  1.89368226, -0.24999131,
                              -0.81189363, -0.03399996,  0.36712684, -0.87035745, -0.67872603,
                              0.94832751,  0.37968557,  0.7727175 ,  0.60546987, -0.64637864,
                              -0.61292911, -0.82198865, -0.67899193, -0.50923558,  3.17272098,
                              -0.87216294,  1.29118515, -0.82366113, -0.86965422,  0.27097461,
                              -0.31773705, -0.01334636,  1.68421708, -0.73083869, -0.53014154,
                              -0.86714551,  0.53857082,  0.04519031, -0.74839969, -0.66644835,
                              -0.04272491,  2.65975463, -0.29438439, -0.41878424,  0.0502175 ,
                              -0.6325518 , -0.63469662, -0.58393577, -0.5639174 ,  0.41483776,
                              -0.65257017, -0.54890362, -0.70261608, -0.68545748, -0.55247833,
                              -0.32941653,  1.90906587,  1.88761762, -0.52245078, -0.60538402,
                              -0.54747374, -0.04987433,  2.07350245, -0.68474254,  0.05736692};

  std::vector<double> dist_matix(N*N);
  correlate::dot::local(dist_matix, data, M, N, 1.0, 0.0, false);

  ASSERT_EQ(dist_matix.size(), 25);
  EXPECT_NEAR(dist_matix[0], 23.99999998, 1e-8);    // D[0,0]
  EXPECT_NEAR(dist_matix[1], 5.33360230, 1e-8);     // D[1,0]
  EXPECT_NEAR(dist_matix[2], 1.23524208, 1e-8);     // D[2,0]
  EXPECT_NEAR(dist_matix[3], 14.30354105, 1e-8);    // D[3,0]
  EXPECT_NEAR(dist_matix[4], 10.17033773, 1e-8);    // D[4,0]
  EXPECT_NEAR(dist_matix[5], 5.33360230, 1e-8);     // D[0,1]
  EXPECT_NEAR(dist_matix[6], 24.00000004, 1e-8);    // D[1,1]
  EXPECT_NEAR(dist_matix[7], 0.11742583, 1e-8);     // D[2,1]
  EXPECT_NEAR(dist_matix[8], 4.50914729, 1e-8);     // D[3,1]
  EXPECT_NEAR(dist_matix[9], 1.85915797, 1e-8);     // D[4,1]
  EXPECT_NEAR(dist_matix[10], 1.23524208, 1e-8);    // D[0,2]
  EXPECT_NEAR(dist_matix[11], 0.11742583, 1e-8);    // D[1,2]
  EXPECT_NEAR(dist_matix[12], 24.00000002, 1e-8);   // D[2,2]
  EXPECT_NEAR(dist_matix[13], 5.43037010, 1e-8);    // D[3,2]
  EXPECT_NEAR(dist_matix[14], 9.48481840, 1e-8);    // D[4,2]
  EXPECT_NEAR(dist_matix[15], 14.30354105, 1e-8);   // D[0,3]
  EXPECT_NEAR(dist_matix[16], 4.50914729, 1e-8);    // D[1,3]
  EXPECT_NEAR(dist_matix[17], 5.43037010, 1e-8);    // D[2,3]
  EXPECT_NEAR(dist_matix[18], 23.99999996, 1e-8);   // D[3,3]
  EXPECT_NEAR(dist_matix[19], 9.22755337, 1e-8);    // D[4,3]
  EXPECT_NEAR(dist_matix[20], 10.17033773, 1e-8);   // D[0,4]
  EXPECT_NEAR(dist_matix[21], 1.85915797, 1e-8);    // D[1,4]
  EXPECT_NEAR(dist_matix[22], 9.48481840, 1e-8);    // D[2,4]
  EXPECT_NEAR(dist_matix[23], 9.22755337, 1e-8);    // D[3,4]
  EXPECT_NEAR(dist_matix[24], 24.00000003, 1e-8);   // D[4,4]
}

TEST(DotLocalDispatch, FullGpuTest) {
#ifdef USE_HIP
  const std::size_t M = 25;
  const std::size_t N = 5;
  std::vector<double> data = {-0.79681066,  0.0499037 , -0.45662631, -0.32100215, -0.03251983,
                              -0.59374907,  0.57441701, -0.68141773, -0.72210498,  3.22695578,
                              -0.58850394, -0.67991912, -0.63795805, -0.72232977, -0.25431403,
                              -0.40642289,  2.08051953,  1.39865222,  0.02742455,  0.02742455,
                              -0.73244538,  1.14388861, -0.51057625,  0.3196534 , -0.71213922,
                              0.07217628, -0.91038683, -0.34190389, -0.60859959, -0.05853971,
                              -0.47612903,  1.09860381, -0.81914883, -0.85432809,  0.28272552,
                              -0.82625486, -0.5542077 ,  0.1774509 , -0.83634368,  3.16023176,
                              1.37933613,  0.02831185,  0.04585762, -0.44191478, -0.39190933,
                              -0.78756644,  1.57233959,  1.37056324, -0.5770172 , -0.70334674,
                              2.02360187,  2.64072002, -0.04699192, -0.7209499 , -0.62513419,
                              -0.76966976, -0.86873346, -0.80864564, -1.00937144,  0.29079906,
                              1.29280406,  0.08292769,  0.36712684, -0.89634138,  0.18686338,
                              -0.53743845,  0.27131112, -0.4887186 ,  1.89368226, -0.24999131,
                              -0.81189363, -0.03399996,  0.36712684, -0.87035745, -0.67872603,
                              0.94832751,  0.37968557,  0.7727175 ,  0.60546987, -0.64637864,
                              -0.61292911, -0.82198865, -0.67899193, -0.50923558,  3.17272098,
                              -0.87216294,  1.29118515, -0.82366113, -0.86965422,  0.27097461,
                              -0.31773705, -0.01334636,  1.68421708, -0.73083869, -0.53014154,
                              -0.86714551,  0.53857082,  0.04519031, -0.74839969, -0.66644835,
                              -0.04272491,  2.65975463, -0.29438439, -0.41878424,  0.0502175 ,
                              -0.6325518 , -0.63469662, -0.58393577, -0.5639174 ,  0.41483776,
                              -0.65257017, -0.54890362, -0.70261608, -0.68545748, -0.55247833,
                              -0.32941653,  1.90906587,  1.88761762, -0.52245078, -0.60538402,
                              -0.54747374, -0.04987433,  2.07350245, -0.68474254,  0.05736692};

  std::vector<double> dist_matix(N*N);
  correlate::dot::local(dist_matix, data, M, N, 1.0, 0.0, true);

  ASSERT_EQ(dist_matix.size(), 25);
  EXPECT_NEAR(dist_matix[0], 23.99999998, 1e-8);    // D[0,0]
  EXPECT_NEAR(dist_matix[1], 5.33360230, 1e-8);     // D[1,0]
  EXPECT_NEAR(dist_matix[2], 1.23524208, 1e-8);     // D[2,0]
  EXPECT_NEAR(dist_matix[3], 14.30354105, 1e-8);    // D[3,0]
  EXPECT_NEAR(dist_matix[4], 10.17033773, 1e-8);    // D[4,0]
  EXPECT_NEAR(dist_matix[5], 5.33360230, 1e-8);     // D[0,1]
  EXPECT_NEAR(dist_matix[6], 24.00000004, 1e-8);    // D[1,1]
  EXPECT_NEAR(dist_matix[7], 0.11742583, 1e-8);     // D[2,1]
  EXPECT_NEAR(dist_matix[8], 4.50914729, 1e-8);     // D[3,1]
  EXPECT_NEAR(dist_matix[9], 1.85915797, 1e-8);     // D[4,1]
  EXPECT_NEAR(dist_matix[10], 1.23524208, 1e-8);    // D[0,2]
  EXPECT_NEAR(dist_matix[11], 0.11742583, 1e-8);    // D[1,2]
  EXPECT_NEAR(dist_matix[12], 24.00000002, 1e-8);   // D[2,2]
  EXPECT_NEAR(dist_matix[13], 5.43037010, 1e-8);    // D[3,2]
  EXPECT_NEAR(dist_matix[14], 9.48481840, 1e-8);    // D[4,2]
  EXPECT_NEAR(dist_matix[15], 14.30354105, 1e-8);   // D[0,3]
  EXPECT_NEAR(dist_matix[16], 4.50914729, 1e-8);    // D[1,3]
  EXPECT_NEAR(dist_matix[17], 5.43037010, 1e-8);    // D[2,3]
  EXPECT_NEAR(dist_matix[18], 23.99999996, 1e-8);   // D[3,3]
  EXPECT_NEAR(dist_matix[19], 9.22755337, 1e-8);    // D[4,3]
  EXPECT_NEAR(dist_matix[20], 10.17033773, 1e-8);   // D[0,4]
  EXPECT_NEAR(dist_matix[21], 1.85915797, 1e-8);    // D[1,4]
  EXPECT_NEAR(dist_matix[22], 9.48481840, 1e-8);    // D[2,4]
  EXPECT_NEAR(dist_matix[23], 9.22755337, 1e-8);    // D[3,4]
  EXPECT_NEAR(dist_matix[24], 24.00000003, 1e-8);   // D[4,4]
#else
  GTEST_SKIP() << "Test skipped: library not built with HIP.";
#endif
}

TEST(DotLocalCorColDispatch, CallsCpu) {
  const std::size_t M = 5, N = 3;
  std::vector<double> out(N, 0.0), X(M*N, 1.0), Y(M*N, 2.0);

  g_cpu_called = false;

  SeamGuard guard_cpu(correlate::dot::seams::local_corr_col_cpu_fn);
  correlate::dot::seams::local_corr_col_cpu_fn = +[](std::vector<double>& out,
                                                     const std::size_t offset,
                                                     const std::vector<double>& X,
                                                     const std::vector<double>& Y,
                                                     const std::size_t M,
                                                     const std::size_t N,
                                                     const double alpha,
                                                     const double beta)
  {
    g_cpu_called = true;
    // ASSERT_EQ(m, M); ASSERT_EQ(n, N);
    // ASSERT_EQ(data.size(), M * N);
    // ASSERT_EQ(a, 1.0); ASSERT_EQ(b, 0.0);
    // sentinel
    out.assign(N * N, 1.23);
  };

  correlate::dot::local_corresponding_columns(out, 0, X, Y, M, N, /*alpha=*/1.0, /*beta=*/0.0 );

  EXPECT_TRUE(g_cpu_called);
  for (double v : out) EXPECT_DOUBLE_EQ(v, 1.23);
}

TEST(DotLocalCorColDispatch, FullCpuTest) {
  const std::size_t offset = 0;
  const std::vector<double> X = {0.774590906, 0.513763385, 0.357532009, 0.487371218, 0.050744795,
                                 0.300630147, 0.880460832, 0.779610085, 0.730755625, 0.008541588,
                                 1.0, 2.0, 5.0, -3.0, -1.0, 50.0, 4.0, -3.0, -1.0, 1.0,
                                 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.0, 0.0,
                                 3.0, -4.0, -5.0, 2.0, 4.0, -5.0, -1.0, 0.0, 0.0, 0.0,
                                 0.17271539, 0.382416903, 0.793810151, 0.560014815, 0.864605157,
                                 0.323185059, 0.744819809, 0.683147407, 0.697989807,0.059140425};

  const std::vector<double> Y = {0.642031426, 0.185417287, 0.514525743, 0.20851762, 0.653119113,
                                 0.277094311, 0.797918211, 0.318373245, 0.568765613, 0.687726279,
                                 -1.0, -2.0, -5.0, 3.0, 1.0, -50.0, -4.0, 3.0, 1.0, -1.0,
                                 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 0.0, 0.0,
                                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                 0.729349489, 0.616345872, 0.368910377, 0.849647312, 0.744862772,
                                 0.45806787, 0.407767465, 0.450712837, 0.689680978, 0.12058077};
  const std::size_t M = 10;
  const std::size_t N = 5;
  const double alpha = 1.0;
  const double beta = 0.0;
  std::vector<double> out(N);
  const std::vector<double> expcected_out = {2.366848195, -2567.0, 20.4, 0.0, 3.022521813};

  correlate::dot::local_corresponding_columns(out, offset, X, Y, M, N, alpha, beta);

  ASSERT_EQ(out.size(), N);
  for (std::size_t i = 0; i < N; ++i) {
    EXPECT_NEAR(out[i], expcected_out[i], 1e-8);
  }
}
