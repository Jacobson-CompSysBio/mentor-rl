/// @file split.hpp
/// @brief Utility functions for partitioning tasks across multiple workers.
///
/// This header provides routines for dividing work (e.g., loop iterations, data shards)
/// among multiple processing units (e.g., MPI ranks, OpenMP threads) in a balanced way.
/// It ensures fair task allocation when the number of tasks is not evenly divisible
/// by the number of workers.
///
/// @author Ken Smith
/// @date 2025-07-28

#pragma once

/// @namespace split
/// @brief Namespace for task-splitting utilites used in distributed or parallel processing.
namespace split {

/// @brief Distributes a fixed number of tasks evenly across multiple workers.
///
/// This function computes the range of task indices `[start_task, stop_task]` that should
/// be assigned to a particular worker, identified by `cur_worker`. If there are more workers
/// than tasks, some workers will be assigned no tasks. In that case, the function returns `false`
/// and does not modify `start_task` or `stop_task`.
///
/// If the number of tasks is not divisible by the number of workers, the tasks are distributed
/// as evenly as possible such that some workers may receive one more task than others.
///
/// @param[in] cur_worker The index of the current worker (0-based).
/// @param[in] num_workers The total number of workers.
/// @param[in] num_tasks The total number of tasks to be distributed.
/// @param[out] start_task The index of the first task assigned to the current worker (inclusive).
/// @param[out] stop_task The index of the last task assigned to the current worker (inclusive).
///
/// @return `true` if the worker was assigned tasks, `false` otherwise.
///
/// @throws std::invalid_argument If `cur_worker >= num_workers`.
bool split_tasks_among_workers(
  const unsigned long cur_worker,
  const unsigned long num_workers,
  const unsigned long num_tasks,
  unsigned long &start_task,
  unsigned long &stop_task
);

} // namespace split
