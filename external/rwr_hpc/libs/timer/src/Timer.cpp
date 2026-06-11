#include "timer/Timer.hpp"

Timer::Timer(const bool start_immediately) :  running(start_immediately),
                                              accumulatedCpuTime(0),
                                              accumulatedWallTime(0),
                                              startCpuTime(start_immediately ? get_cpu_time() : 0),
                                              startWallTime(start_immediately ? get_wall_time() : 0) {}


// tm* Timer::current_time() const {
//   time_t rawtime;
//   time(&rawtime);
//   return localtime(&rawtime);
// }

double Timer::elapsed_cpu_time() const {
  if (running) {
    return (get_cpu_time() - startCpuTime + accumulatedCpuTime) / CLOCKS_PER_SEC;
  }

  return accumulatedCpuTime / CLOCKS_PER_SEC;
}

double Timer::elapsed_wall_time() const {
  if (running) {
    return get_wall_time() - startWallTime + accumulatedWallTime;
  }

  return accumulatedWallTime;
}

double Timer::get_cpu_time() const {
  return clock();
}

double Timer::get_wall_time() const {
  struct timeval time;
  gettimeofday(&time, NULL);
  return time.tv_sec + time.tv_usec * 0.000001;
}

void Timer::reset() {
  running = false;
  accumulatedCpuTime = 0;
  accumulatedWallTime = 0;
}

void Timer::restart() {
  startCpuTime = get_cpu_time();
  startWallTime = get_wall_time();
  accumulatedCpuTime = 0;
  accumulatedWallTime = 0;
  running = true;
}

void Timer::start() {
  if (!running) {
    startCpuTime = get_cpu_time();
    startWallTime = get_wall_time();
    running = true;
  }
}

void Timer::stop() {
  if (running) {
    accumulatedCpuTime += get_cpu_time() - startCpuTime;
    accumulatedWallTime += get_wall_time() - startWallTime;
    running = false;
  }
}
