/// @file Timer.hpp
/// @brief A utility class for measuring CPU and wall-clock time.
///
/// The Timer class provides functionality for tracking elapsed time,
/// including CPU time and real (wall) time. It can be started, stopped,
/// reset, and restarted, and it supports both cumulative and interval timing.

#pragma once

#include <stdexcept>
#include <sys/time.h>
#include <time.h>

/// @class Timer
/// @brief Utility class to measure elapsed CPU and wall-clock time.
///
/// This class supports start/stop timing as well as cumulative timing across intervals.
/// It uses `clock()` to measure CPU time and `gettimeofday()` for wall time in seconds.
///
/// @note This timer is not thread-safe.
class Timer {
public:
  /// @brief Constructs a Timer and optionally starts it immediately.
  /// @param start_immediately If true, the timer starts upon construction. (Default: false).
  Timer(const bool start_immediately = false);

  // /// @brief Returns the current system time as a `tm` structure (local time).
  // ///
  // /// @return A pointer to a `tm` struct containing the current local time.
  // ///
  // tm *current_time() const;

  /// @brief Returns the total elapsed CPU time in seconds.
  ///
  /// @return Elapsed CPU time since start or since last reset.
  double elapsed_cpu_time() const;

  /// @brief Returns the total elapsed wall-clock time in seconds.
  ///
  /// @return Elapsed real time since start or since last reset.
  double elapsed_wall_time() const;

  /// @brief Resets the timer to zero and stops it.
  void reset();

  /// @brief Resets the timer and starts it.
  void restart();

  /// @brief Starts or resumes the timer.
  void start();

  /// @brief Stops the timer and accumulates time since last start.
  void stop();

private:
  bool running;               ///< True if the timer is currently running
  double accumulatedCpuTime;  ///< Total accumulated CPU time (in clock ticks)
  double accumulatedWallTime; ///< Total accumulated wall time (in seconds)
  double startCpuTime;        ///< Start CPU time of the current timing interval
  double startWallTime;       ///< Start wall time of the current timing interval

  /// @brief Returns the current CPU time using `clock()`.
  ///
  /// @return CPU time in clock ticks.
  double get_cpu_time() const;

  /// @brief Returns the current wall time using `gettimeofday()`.
  ///
  /// @return Wall-clock time in seconds (with microsecond resolution).
  double get_wall_time() const;
};
