// tests/test_spearman_local_dispatch.cpp
#include <gtest/gtest.h>

#include <correlate/spearman.hpp>
#include "metrics/spearman/spearman_seams.hpp"

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
TEST(SpearmanLocalDispatch, CallsCpuWhenUseGpuFalse) {
  const std::size_t M = 5, N = 3;
  std::vector<double> out(N*N, 0.0), data(M*N, 1.0);

  g_cpu_called = false;
  g_gpu_called = false;
  g_hip_avail_called = false;

  SeamGuard guard_cpu(correlate::spearman::seams::local_cpu_fn);
  correlate::spearman::seams::local_cpu_fn = +[](std::vector<double>& out, const std::vector<double>&,
                                                std::size_t, std::size_t)
  {
    g_cpu_called = true;
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

  SeamGuard guard_gpu(correlate::spearman::seams::local_gpu_fn);
  correlate::spearman::seams::local_gpu_fn = +[](std::vector<double>& out, const std::vector<double>&,
                                                std::size_t, std::size_t)
  {
    g_gpu_called = true;
    out.assign(N * N, 9.99); // should not be used in this test
  };
#endif
  
  correlate::spearman::local(out, data, M, N, /*use_gpu=*/false);

  EXPECT_TRUE(g_cpu_called);
#ifdef USE_HIP
  EXPECT_FALSE(g_gpu_called);
  EXPECT_FALSE(g_hip_avail_called);
#endif
  for (double v : out) EXPECT_DOUBLE_EQ(v, 1.23);
}

TEST(SpearmanLocalDispatch, CpuFallbackWhenGpuRequestedButUnavailableOrDisabled) {
  const std::size_t M = 5, N = 3;
  std::vector<double> out(N*N, 0.0), data(M*N, 1.0);

  g_cpu_called = false;
  g_gpu_called = false;
  g_hip_avail_called = false;

  SeamGuard guard_cpu(correlate::spearman::seams::local_cpu_fn);
  correlate::spearman::seams::local_cpu_fn = +[](std::vector<double>& out, const std::vector<double>&,
                                                std::size_t, std::size_t)
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

  SeamGuard guard_gpu(correlate::spearman::seams::local_gpu_fn);
  correlate::spearman::seams::local_gpu_fn = +[](std::vector<double>& out, const std::vector<double>&,
                                                std::size_t, std::size_t)
  {
    g_gpu_called = true;
    out.assign(N * N, 9.99); // should not be used in this test
  };
#endif

  correlate::spearman::local(out, data, M, N, /*use_gpu=*/true);

  EXPECT_TRUE(g_cpu_called);
#ifdef USE_HIP
  EXPECT_FALSE(g_gpu_called);
  EXPECT_TRUE(g_hip_avail_called);
#endif
  for (double v : out) EXPECT_DOUBLE_EQ(v, 1.23);
}

TEST(SpearmanLocalDispatch, GpuPathWhenGpuRequestedAndHipAvailable) {
#ifdef USE_HIP
  const std::size_t M = 5, N = 3;
  std::vector<double> out(N*N, 0.0), data(M*N, 1.0);

  g_cpu_called = false;
  g_gpu_called = false;
  g_hip_avail_called = false;

  SeamGuard guard_cpu(correlate::spearman::seams::local_cpu_fn);
  correlate::spearman::seams::local_cpu_fn = +[](std::vector<double>& out, const std::vector<double>&,
                                                std::size_t, std::size_t)
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

  SeamGuard guard_gpu(correlate::spearman::seams::local_gpu_fn);
  correlate::spearman::seams::local_gpu_fn = +[](std::vector<double>& out, const std::vector<double>&,
                                                std::size_t, std::size_t)
  {
    g_gpu_called = true;
    out.assign(N * N, 9.99); // should not be used in this test
  };
#endif

  correlate::spearman::local(out, data, M, N, /*use_gpu=*/true);

  EXPECT_FALSE(g_cpu_called);
  EXPECT_TRUE(g_gpu_called);
  EXPECT_TRUE(g_hip_avail_called);
  for (double v : out) EXPECT_DOUBLE_EQ(v, 9.99);
#else
  GTEST_SKIP() << "Test skipped: library not built with HIP.";
#endif
}

TEST(SpearmanLocalDispatch, FullCpuTest) {
  const std::size_t M = 25;
  const std::size_t N = 5;
  const std::vector<double> data = {-0.673302386, 1.10479777, 0.041084749, 0.325895485, 0.931708357,
                              -0.246873047, 2.206275721, -0.430977233, -0.516420458, 7.776607138,
                              -0.235858274, -0.427830152,-0.339711905,-0.516892517,0.465940537,
                              0.146511931, 5.369091013, 3.937169662, 1.057591555, 1.057591555,
                              -0.538135298, 3.402166081, -0.072210125, 1.67127214, -0.49549236,
                              2.158787816, -0.002851026, 1.247811442, 0.661080902, 1.871212638,
                              0.952516134, 4.416928382, 0.197872574, 0.120478202, 2.621996144,
                              0.182239308, 0.78074306, 2.39039198, 0.160043904, 8.952509872,
                              5.034539486, 2.06228607, 2.100886764, 1.027787484, 1.137799474, 
                              0.267353832, 5.459147098, 5.015239128, 0.73056216, 0.452637172,
                              7.654284301, 9.073656046, 2.891918584, 1.34181523, 1.562191363,
                              1.229759552, 1.001913042, 1.140115028, 0.678445688, 3.668837838,
                              5.973449338, 3.190733687, 3.844391732, 0.938414826, 3.429785774,
                              1.763891565, 3.624015576, 1.87594722, 7.355469198, 2.425019987,
                              1.132644651, 2.921800092, 3.844391732, 0.998177865, 1.438930131,
                              6.181153273, 4.873276811, 5.77725025, 5.392580701, 2.513329128,
                              2.590263047, 2.109426105, 2.438318561, 2.828758166, 11.29725825,
                              1.994025238, 6.969725845, 2.105579401, 1.999795294, 4.623241603,
                              3.269204785, 3.969303372, 7.873699284, 2.319071013, 2.780674458,
                              2.005565327, 5.238712886, 4.103937713, 2.278680713, 2.467168795,
                              4.893187725, 11.64938658, 4.264039025, 3.9530394, 5.12554375,
                              3.4186205, 3.41325845, 3.540160575, 3.5902065, 6.0370944,
                              3.368574575, 3.62774095, 3.2434598, 3.2863563, 3.618804175,
                              4.176458675, 9.772664675, 9.71904405, 3.69387305, 3.48653995,
                              3.63131565, 4.875314175, 10.18375613, 3.28814365, 5.1434173};

  std::vector<double> dist_matix(N*N);
  correlate::spearman::local(dist_matix, data, M, N, false);

  ASSERT_EQ(dist_matix.size(), 25);
  EXPECT_NEAR(dist_matix[0], 1.0, 1e-8);                  // D[0,0]
  EXPECT_NEAR(dist_matix[1], 0.4343142992438976, 1e-8);   // D[1,0]
  EXPECT_NEAR(dist_matix[2], 0.21893035782993456, 1e-8);  // D[2,0]
  EXPECT_NEAR(dist_matix[3], 0.3196768668482541, 1e-8);   // D[3,0]
  EXPECT_NEAR(dist_matix[4], 0.33391037355509573, 1e-8);  // D[4,0]

  EXPECT_NEAR(dist_matix[5], 0.4343142992438976, 1e-8);   // D[0,1]
  EXPECT_NEAR(dist_matix[6], 1.0, 1e-8);                  // D[1,1]
  EXPECT_NEAR(dist_matix[7], 0.3112137006982401, 1e-8);   // D[2,1]
  EXPECT_NEAR(dist_matix[8], 0.36846153846153845, 1e-8);  // D[3,1]
  EXPECT_NEAR(dist_matix[9], 0.23384615384615384, 1e-8);  // D[4,1]

  EXPECT_NEAR(dist_matix[10], 0.21893035782993456, 1e-8); // D[0,2]
  EXPECT_NEAR(dist_matix[11], 0.3112137006982401, 1e-8);  // D[1,2]
  EXPECT_NEAR(dist_matix[12], 1.0, 1e-8);                 // D[2,2]
  EXPECT_NEAR(dist_matix[13], 0.3546835995596754, 1e-8);  // D[3,2]
  EXPECT_NEAR(dist_matix[14], 0.43354492050298726, 1e-8); // D[4,2]

  EXPECT_NEAR(dist_matix[15], 0.3196768668482541, 1e-8);  // D[0,3]
  EXPECT_NEAR(dist_matix[16], 0.36846153846153845, 1e-8); // D[1,3]
  EXPECT_NEAR(dist_matix[17], 0.3546835995596754, 1e-8);  // D[2,3]
  EXPECT_NEAR(dist_matix[18], 1.0, 1e-8);                 // D[3,3]
  EXPECT_NEAR(dist_matix[19], 0.66, 1e-8);                // D[4,3]

  EXPECT_NEAR(dist_matix[20], 0.33391037355509573, 1e-8); // D[0,4]
  EXPECT_NEAR(dist_matix[21], 0.23384615384615384, 1e-8); // D[1,4]
  EXPECT_NEAR(dist_matix[22], 0.43354492050298726, 1e-8); // D[2,4]
  EXPECT_NEAR(dist_matix[23], 0.66, 1e-8);                // D[3,4]
  EXPECT_NEAR(dist_matix[24], 1.0, 1e-8);                 // D[4,4]
}

TEST(SpearmanLocalDispatch, FullGpuTest) {
#ifdef USE_HIP
  const std::size_t M = 25;
  const std::size_t N = 5;
  const std::vector<double> data = {-0.673302386, 1.10479777, 0.041084749, 0.325895485, 0.931708357,
                              -0.246873047, 2.206275721, -0.430977233, -0.516420458, 7.776607138,
                              -0.235858274, -0.427830152,-0.339711905,-0.516892517,0.465940537,
                              0.146511931, 5.369091013, 3.937169662, 1.057591555, 1.057591555,
                              -0.538135298, 3.402166081, -0.072210125, 1.67127214, -0.49549236,
                              2.158787816, -0.002851026, 1.247811442, 0.661080902, 1.871212638,
                              0.952516134, 4.416928382, 0.197872574, 0.120478202, 2.621996144,
                              0.182239308, 0.78074306, 2.39039198, 0.160043904, 8.952509872,
                              5.034539486, 2.06228607, 2.100886764, 1.027787484, 1.137799474, 
                              0.267353832, 5.459147098, 5.015239128, 0.73056216, 0.452637172,
                              7.654284301, 9.073656046, 2.891918584, 1.34181523, 1.562191363,
                              1.229759552, 1.001913042, 1.140115028, 0.678445688, 3.668837838,
                              5.973449338, 3.190733687, 3.844391732, 0.938414826, 3.429785774,
                              1.763891565, 3.624015576, 1.87594722, 7.355469198, 2.425019987,
                              1.132644651, 2.921800092, 3.844391732, 0.998177865, 1.438930131,
                              6.181153273, 4.873276811, 5.77725025, 5.392580701, 2.513329128,
                              2.590263047, 2.109426105, 2.438318561, 2.828758166, 11.29725825,
                              1.994025238, 6.969725845, 2.105579401, 1.999795294, 4.623241603,
                              3.269204785, 3.969303372, 7.873699284, 2.319071013, 2.780674458,
                              2.005565327, 5.238712886, 4.103937713, 2.278680713, 2.467168795,
                              4.893187725, 11.64938658, 4.264039025, 3.9530394, 5.12554375,
                              3.4186205, 3.41325845, 3.540160575, 3.5902065, 6.0370944,
                              3.368574575, 3.62774095, 3.2434598, 3.2863563, 3.618804175,
                              4.176458675, 9.772664675, 9.71904405, 3.69387305, 3.48653995,
                              3.63131565, 4.875314175, 10.18375613, 3.28814365, 5.1434173};

  std::vector<double> dist_matix(N*N);
  correlate::spearman::local(dist_matix, data, M, N, true);

  ASSERT_EQ(dist_matix.size(), 25);
  EXPECT_NEAR(dist_matix[0], 1.0, 1e-8);                  // D[0,0]
  EXPECT_NEAR(dist_matix[1], 0.4343142992438976, 1e-8);   // D[1,0]
  EXPECT_NEAR(dist_matix[2], 0.21893035782993456, 1e-8);  // D[2,0]
  EXPECT_NEAR(dist_matix[3], 0.3196768668482541, 1e-8);   // D[3,0]
  EXPECT_NEAR(dist_matix[4], 0.33391037355509573, 1e-8);  // D[4,0]

  EXPECT_NEAR(dist_matix[5], 0.4343142992438976, 1e-8);   // D[0,1]
  EXPECT_NEAR(dist_matix[6], 1.0, 1e-8);                  // D[1,1]
  EXPECT_NEAR(dist_matix[7], 0.3112137006982401, 1e-8);   // D[2,1]
  EXPECT_NEAR(dist_matix[8], 0.36846153846153845, 1e-8);  // D[3,1]
  EXPECT_NEAR(dist_matix[9], 0.23384615384615384, 1e-8);  // D[4,1]

  EXPECT_NEAR(dist_matix[10], 0.21893035782993456, 1e-8); // D[0,2]
  EXPECT_NEAR(dist_matix[11], 0.3112137006982401, 1e-8);  // D[1,2]
  EXPECT_NEAR(dist_matix[12], 1.0, 1e-8);                 // D[2,2]
  EXPECT_NEAR(dist_matix[13], 0.3546835995596754, 1e-8);  // D[3,2]
  EXPECT_NEAR(dist_matix[14], 0.43354492050298726, 1e-8); // D[4,2]

  EXPECT_NEAR(dist_matix[15], 0.3196768668482541, 1e-8);  // D[0,3]
  EXPECT_NEAR(dist_matix[16], 0.36846153846153845, 1e-8); // D[1,3]
  EXPECT_NEAR(dist_matix[17], 0.3546835995596754, 1e-8);  // D[2,3]
  EXPECT_NEAR(dist_matix[18], 1.0, 1e-8);                 // D[3,3]
  EXPECT_NEAR(dist_matix[19], 0.66, 1e-8);                // D[4,3]

  EXPECT_NEAR(dist_matix[20], 0.33391037355509573, 1e-8); // D[0,4]
  EXPECT_NEAR(dist_matix[21], 0.23384615384615384, 1e-8); // D[1,4]
  EXPECT_NEAR(dist_matix[22], 0.43354492050298726, 1e-8); // D[2,4]
  EXPECT_NEAR(dist_matix[23], 0.66, 1e-8);                // D[3,4]
  EXPECT_NEAR(dist_matix[24], 1.0, 1e-8);                 // D[4,4]
#else
  GTEST_SKIP() << "Test skipped: library not built with HIP.";
#endif
}

TEST(SpearmanLocalDistanceDispatch, CallsCpuWhenUseGpuFalse) {
  const std::size_t M = 5, N = 3;
  std::vector<double> out(N*N, 0.0), data(M*N, 1.0);

  g_cpu_called = false;
  g_gpu_called = false;
  g_hip_avail_called = false;

  SeamGuard guard_cpu(correlate::spearman::seams::local_distance_cpu_fn);
  correlate::spearman::seams::local_distance_cpu_fn = +[](std::vector<double>& out, const std::vector<double>&,
                                                std::size_t, std::size_t)
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

  SeamGuard guard_gpu(correlate::spearman::seams::local_distance_gpu_fn);
  correlate::spearman::seams::local_distance_gpu_fn = +[](std::vector<double>& out, const std::vector<double>&,
                                                std::size_t, std::size_t)
  {
    g_gpu_called = true;
    out.assign(N * N, 9.99); // should not be used in this test
  };
#endif
  
  correlate::spearman::local_distance(out, data, M, N, /*use_gpu=*/false);

  EXPECT_TRUE(g_cpu_called);
#ifdef USE_HIP
  EXPECT_FALSE(g_gpu_called);
  EXPECT_FALSE(g_hip_avail_called);
#endif
  for (double v : out) EXPECT_DOUBLE_EQ(v, 1.23);
}

TEST(SpearmanLocalDistanceDispatch, CpuFallbackWhenGpuRequestedButUnavailableOrDisabled) {
  const std::size_t M = 5, N = 3;
  std::vector<double> out(N*N, 0.0), data(M*N, 1.0);

  g_cpu_called = false;
  g_gpu_called = false;
  g_hip_avail_called = false;

  SeamGuard guard_cpu(correlate::spearman::seams::local_distance_cpu_fn);
  correlate::spearman::seams::local_distance_cpu_fn = +[](std::vector<double>& out, const std::vector<double>&,
                                                std::size_t, std::size_t)
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

  SeamGuard guard_gpu(correlate::spearman::seams::local_distance_gpu_fn);
  correlate::spearman::seams::local_distance_gpu_fn = +[](std::vector<double>& out, const std::vector<double>&,
                                                std::size_t, std::size_t)
  {
    g_gpu_called = true;
    out.assign(N * N, 9.99); // should not be used in this test
  };
#endif

  correlate::spearman::local_distance(out, data, M, N, /*use_gpu=*/true);

  EXPECT_TRUE(g_cpu_called);
#ifdef USE_HIP
  EXPECT_FALSE(g_gpu_called);
  EXPECT_TRUE(g_hip_avail_called);
#endif
  for (double v : out) EXPECT_DOUBLE_EQ(v, 1.23);
}

TEST(SpearmanLocalDistanceDispatch, GpuPathWhenGpuRequestedAndHipAvailable) {
#ifdef USE_HIP
  const std::size_t M = 5, N = 3;
  std::vector<double> out(N*N, 0.0), data(M*N, 1.0);

  g_cpu_called = false;
  g_gpu_called = false;
  g_hip_avail_called = false;

  SeamGuard guard_cpu(correlate::spearman::seams::local_distance_cpu_fn);
  correlate::spearman::seams::local_distance_cpu_fn = +[](std::vector<double>& out, const std::vector<double>&,
                                                std::size_t, std::size_t)
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

  SeamGuard guard_gpu(correlate::spearman::seams::local_distance_gpu_fn);
  correlate::spearman::seams::local_distance_gpu_fn = +[](std::vector<double>& out, const std::vector<double>&,
                                                std::size_t, std::size_t)
  {
    g_gpu_called = true;
    out.assign(N * N, 9.99); // should not be used in this test
  };
#endif

  correlate::spearman::local_distance(out, data, M, N, /*use_gpu=*/true);

  EXPECT_FALSE(g_cpu_called);
  EXPECT_TRUE(g_gpu_called);
  EXPECT_TRUE(g_hip_avail_called);
  for (double v : out) EXPECT_DOUBLE_EQ(v, 9.99);
#else
  GTEST_SKIP() << "Test skipped: library not built with HIP.";
#endif
}

TEST(SpearmanLocalDistanceDispatch, FullCpuTest) {
  const std::size_t M = 25;
  const std::size_t N = 5;
  const std::vector<double> data = {-0.673302386, 1.10479777, 0.041084749, 0.325895485, 0.931708357,
                              -0.246873047, 2.206275721, -0.430977233, -0.516420458, 7.776607138,
                              -0.235858274, -0.427830152,-0.339711905,-0.516892517,0.465940537,
                              0.146511931, 5.369091013, 3.937169662, 1.057591555, 1.057591555,
                              -0.538135298, 3.402166081, -0.072210125, 1.67127214, -0.49549236,
                              2.158787816, -0.002851026, 1.247811442, 0.661080902, 1.871212638,
                              0.952516134, 4.416928382, 0.197872574, 0.120478202, 2.621996144,
                              0.182239308, 0.78074306, 2.39039198, 0.160043904, 8.952509872,
                              5.034539486, 2.06228607, 2.100886764, 1.027787484, 1.137799474, 
                              0.267353832, 5.459147098, 5.015239128, 0.73056216, 0.452637172,
                              7.654284301, 9.073656046, 2.891918584, 1.34181523, 1.562191363,
                              1.229759552, 1.001913042, 1.140115028, 0.678445688, 3.668837838,
                              5.973449338, 3.190733687, 3.844391732, 0.938414826, 3.429785774,
                              1.763891565, 3.624015576, 1.87594722, 7.355469198, 2.425019987,
                              1.132644651, 2.921800092, 3.844391732, 0.998177865, 1.438930131,
                              6.181153273, 4.873276811, 5.77725025, 5.392580701, 2.513329128,
                              2.590263047, 2.109426105, 2.438318561, 2.828758166, 11.29725825,
                              1.994025238, 6.969725845, 2.105579401, 1.999795294, 4.623241603,
                              3.269204785, 3.969303372, 7.873699284, 2.319071013, 2.780674458,
                              2.005565327, 5.238712886, 4.103937713, 2.278680713, 2.467168795,
                              4.893187725, 11.64938658, 4.264039025, 3.9530394, 5.12554375,
                              3.4186205, 3.41325845, 3.540160575, 3.5902065, 6.0370944,
                              3.368574575, 3.62774095, 3.2434598, 3.2863563, 3.618804175,
                              4.176458675, 9.772664675, 9.71904405, 3.69387305, 3.48653995,
                              3.63131565, 4.875314175, 10.18375613, 3.28814365, 5.1434173};

  std::vector<double> dist_matix(N*N);
  correlate::spearman::local_distance(dist_matix, data, M, N, false);

  ASSERT_EQ(dist_matix.size(), 25);
  EXPECT_NEAR(dist_matix[0], 0.0, 1e-8);                      // D[0,0]
  EXPECT_NEAR(dist_matix[1], 1.0-0.4343142992438976, 1e-8);   // D[1,0]
  EXPECT_NEAR(dist_matix[2], 1.0-0.21893035782993456, 1e-8);  // D[2,0]
  EXPECT_NEAR(dist_matix[3], 1.0-0.3196768668482541, 1e-8);   // D[3,0]
  EXPECT_NEAR(dist_matix[4], 1.0-0.33391037355509573, 1e-8);  // D[4,0]

  EXPECT_NEAR(dist_matix[5], 1.0-0.4343142992438976, 1e-8);   // D[0,1]
  EXPECT_NEAR(dist_matix[6], 0.0, 1e-8);                      // D[1,1]
  EXPECT_NEAR(dist_matix[7], 1.0-0.3112137006982401, 1e-8);   // D[2,1]
  EXPECT_NEAR(dist_matix[8], 1.0-0.36846153846153845, 1e-8);  // D[3,1]
  EXPECT_NEAR(dist_matix[9], 1.0-0.23384615384615384, 1e-8);  // D[4,1]

  EXPECT_NEAR(dist_matix[10], 1.0-0.21893035782993456, 1e-8); // D[0,2]
  EXPECT_NEAR(dist_matix[11], 1.0-0.3112137006982401, 1e-8);  // D[1,2]
  EXPECT_NEAR(dist_matix[12], 0.0, 1e-8);                     // D[2,2]
  EXPECT_NEAR(dist_matix[13], 1.0-0.3546835995596754, 1e-8);  // D[3,2]
  EXPECT_NEAR(dist_matix[14], 1.0-0.43354492050298726, 1e-8); // D[4,2]

  EXPECT_NEAR(dist_matix[15], 1.0-0.3196768668482541, 1e-8);  // D[0,3]
  EXPECT_NEAR(dist_matix[16], 1.0-0.36846153846153845, 1e-8); // D[1,3]
  EXPECT_NEAR(dist_matix[17], 1.0-0.3546835995596754, 1e-8);  // D[2,3]
  EXPECT_NEAR(dist_matix[18], 0.0, 1e-8);                     // D[3,3]
  EXPECT_NEAR(dist_matix[19], 1.0-0.66, 1e-8);                // D[4,3]

  EXPECT_NEAR(dist_matix[20], 1.0-0.33391037355509573, 1e-8); // D[0,4]
  EXPECT_NEAR(dist_matix[21], 1.0-0.23384615384615384, 1e-8); // D[1,4]
  EXPECT_NEAR(dist_matix[22], 1.0-0.43354492050298726, 1e-8); // D[2,4]
  EXPECT_NEAR(dist_matix[23], 1.0-0.66, 1e-8);                // D[3,4]
  EXPECT_NEAR(dist_matix[24], 0.0, 1e-8);                     // D[4,4]
}

TEST(SpearmanLocalDistanceDispatch, FullGpuTest) {
  #ifdef USE_HIP
  const std::size_t M = 25;
  const std::size_t N = 5;
  const std::vector<double> data = {-0.673302386, 1.10479777, 0.041084749, 0.325895485, 0.931708357,
                              -0.246873047, 2.206275721, -0.430977233, -0.516420458, 7.776607138,
                              -0.235858274, -0.427830152,-0.339711905,-0.516892517,0.465940537,
                              0.146511931, 5.369091013, 3.937169662, 1.057591555, 1.057591555,
                              -0.538135298, 3.402166081, -0.072210125, 1.67127214, -0.49549236,
                              2.158787816, -0.002851026, 1.247811442, 0.661080902, 1.871212638,
                              0.952516134, 4.416928382, 0.197872574, 0.120478202, 2.621996144,
                              0.182239308, 0.78074306, 2.39039198, 0.160043904, 8.952509872,
                              5.034539486, 2.06228607, 2.100886764, 1.027787484, 1.137799474, 
                              0.267353832, 5.459147098, 5.015239128, 0.73056216, 0.452637172,
                              7.654284301, 9.073656046, 2.891918584, 1.34181523, 1.562191363,
                              1.229759552, 1.001913042, 1.140115028, 0.678445688, 3.668837838,
                              5.973449338, 3.190733687, 3.844391732, 0.938414826, 3.429785774,
                              1.763891565, 3.624015576, 1.87594722, 7.355469198, 2.425019987,
                              1.132644651, 2.921800092, 3.844391732, 0.998177865, 1.438930131,
                              6.181153273, 4.873276811, 5.77725025, 5.392580701, 2.513329128,
                              2.590263047, 2.109426105, 2.438318561, 2.828758166, 11.29725825,
                              1.994025238, 6.969725845, 2.105579401, 1.999795294, 4.623241603,
                              3.269204785, 3.969303372, 7.873699284, 2.319071013, 2.780674458,
                              2.005565327, 5.238712886, 4.103937713, 2.278680713, 2.467168795,
                              4.893187725, 11.64938658, 4.264039025, 3.9530394, 5.12554375,
                              3.4186205, 3.41325845, 3.540160575, 3.5902065, 6.0370944,
                              3.368574575, 3.62774095, 3.2434598, 3.2863563, 3.618804175,
                              4.176458675, 9.772664675, 9.71904405, 3.69387305, 3.48653995,
                              3.63131565, 4.875314175, 10.18375613, 3.28814365, 5.1434173};

  std::vector<double> dist_matix(N*N);
  correlate::spearman::local_distance(dist_matix, data, M, N, true);

  ASSERT_EQ(dist_matix.size(), 25);
  EXPECT_NEAR(dist_matix[0], 0.0, 1e-8);                  // D[0,0]
  EXPECT_NEAR(dist_matix[1], 1.0-0.4343142992438976, 1e-8);   // D[1,0]
  EXPECT_NEAR(dist_matix[2], 1.0-0.21893035782993456, 1e-8);  // D[2,0]
  EXPECT_NEAR(dist_matix[3], 1.0-0.3196768668482541, 1e-8);   // D[3,0]
  EXPECT_NEAR(dist_matix[4], 1.0-0.33391037355509573, 1e-8);  // D[4,0]

  EXPECT_NEAR(dist_matix[5], 1.0-0.4343142992438976, 1e-8);   // D[0,1]
  EXPECT_NEAR(dist_matix[6], 0.0, 1e-8);                  // D[1,1]
  EXPECT_NEAR(dist_matix[7], 1.0-0.3112137006982401, 1e-8);   // D[2,1]
  EXPECT_NEAR(dist_matix[8], 1.0-0.36846153846153845, 1e-8);  // D[3,1]
  EXPECT_NEAR(dist_matix[9], 1.0-0.23384615384615384, 1e-8);  // D[4,1]

  EXPECT_NEAR(dist_matix[10], 1.0-0.21893035782993456, 1e-8); // D[0,2]
  EXPECT_NEAR(dist_matix[11], 1.0-0.3112137006982401, 1e-8);  // D[1,2]
  EXPECT_NEAR(dist_matix[12], 0.0, 1e-8);                 // D[2,2]
  EXPECT_NEAR(dist_matix[13], 1.0-0.3546835995596754, 1e-8);  // D[3,2]
  EXPECT_NEAR(dist_matix[14], 1.0-0.43354492050298726, 1e-8); // D[4,2]

  EXPECT_NEAR(dist_matix[15], 1.0-0.3196768668482541, 1e-8);  // D[0,3]
  EXPECT_NEAR(dist_matix[16], 1.0-0.36846153846153845, 1e-8); // D[1,3]
  EXPECT_NEAR(dist_matix[17], 1.0-0.3546835995596754, 1e-8);  // D[2,3]
  EXPECT_NEAR(dist_matix[18], 0.0, 1e-8);                 // D[3,3]
  EXPECT_NEAR(dist_matix[19], 1.0-0.66, 1e-8);                // D[4,3]

  EXPECT_NEAR(dist_matix[20], 1.0-0.33391037355509573, 1e-8); // D[0,4]
  EXPECT_NEAR(dist_matix[21], 1.0-0.23384615384615384, 1e-8); // D[1,4]
  EXPECT_NEAR(dist_matix[22], 1.0-0.43354492050298726, 1e-8); // D[2,4]
  EXPECT_NEAR(dist_matix[23], 1.0-0.66, 1e-8);                // D[3,4]
  EXPECT_NEAR(dist_matix[24], 0.0, 1e-8);                 // D[4,4]
#else
  GTEST_SKIP() << "Test skipped: library not built with HIP.";
#endif
}