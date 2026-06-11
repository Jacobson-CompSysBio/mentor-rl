#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <timer/Timer.hpp>

#include <chrono>
#include <thread>
using namespace std::chrono_literals;

TEST(TimerTest, TimesInitializeToZero) {
  Timer timer;

  EXPECT_EQ(timer.elapsed_cpu_time(), 0.0);
  EXPECT_EQ(timer.elapsed_wall_time(), 0.0);
}

TEST(TimTimerTester, TimesStartImmediately) {
  Timer timer(true);

  EXPECT_NE(timer.elapsed_cpu_time(), 0.0);
  EXPECT_NE(timer.elapsed_wall_time(), 0.0);
}

TEST(TimerTest, StartsCorrectly) {
  Timer timer;

  timer.start();

  std::this_thread::sleep_for(1ms);

  EXPECT_NE(timer.elapsed_cpu_time(), 0.0);
  EXPECT_NE(timer.elapsed_wall_time(), 0.0);
}

TEST(TimerTest, StopsCorrectly) {
  Timer timer;

  timer.start();
  std::this_thread::sleep_for(1ms);
  timer.stop();

  double expected_cpu_time = timer.elapsed_cpu_time();
  double expected_wall_time = timer.elapsed_wall_time();

  std::this_thread::sleep_for(1ms);

  EXPECT_DOUBLE_EQ(timer.elapsed_cpu_time(), expected_cpu_time);
  EXPECT_DOUBLE_EQ(timer.elapsed_wall_time(), expected_wall_time);
}

TEST(TimerTest, RestartsCorrectly) {
  Timer timer;

  timer.start();
  std::this_thread::sleep_for(100ms);
  double first_cpu_time = timer.elapsed_cpu_time();
  double first_wall_time = timer.elapsed_wall_time();

  timer.restart();
  double second_cpu_time = timer.elapsed_cpu_time();
  double second_wall_time = timer.elapsed_wall_time();

  EXPECT_LT(second_cpu_time, first_cpu_time);
  EXPECT_LT(second_wall_time, first_wall_time);
}

TEST(TimerTest, ResestsCorrectly) {
  Timer timer;

  timer.start();
  std::this_thread::sleep_for(100ms);
  double first_cpu_time = timer.elapsed_cpu_time();
  double first_wall_time = timer.elapsed_wall_time();

  timer.reset();
  EXPECT_NE(first_cpu_time, 0.0);
  EXPECT_NE(first_wall_time, 0.0);
  EXPECT_DOUBLE_EQ(timer.elapsed_cpu_time(), 0.0);
  EXPECT_DOUBLE_EQ(timer.elapsed_wall_time(), 0.0);
}
