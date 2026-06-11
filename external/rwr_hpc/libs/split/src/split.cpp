#include "split/split.hpp"
#include <cmath>
#include <stdexcept>

namespace split {

bool split_tasks_among_workers(
  const unsigned long cur_worker,
  const unsigned long num_workers,
  const unsigned long num_tasks,
  unsigned long &start_task,
  unsigned long &stop_task)
{
  // Check that cur_worker is less than num_workes
  if (cur_worker >= num_workers) {
    throw std::invalid_argument("split_tasks_among_workers - cur_worker must be less than num_workers");
  }

  // Return false (worker is assigned no tasks)
  if (cur_worker >= num_tasks) {
    return false;
  }
  
  // The tasks will be split amount workers such that
  // n = (a * n1) + (b * n2), where n2 = n1 - 1

  // Determine the number of tasks to assign per worker
  double avg_tasks_per_worker = static_cast<double>(num_tasks) / num_workers;
  unsigned long n1 = static_cast<unsigned long>(std::ceil(avg_tasks_per_worker));
  unsigned long n2 = n1 - 1;

  // Calculate the number of workes assinged n1 tasks and the number of
  // workers assigned n2 tasks
  unsigned long max_tasks = n1 * num_workers;
  unsigned long b = max_tasks - num_tasks;
  unsigned long a = num_workers - b;

  // Assign all workers before 'a' n1 tasks and all workers at or after
  // 'a' n2 tasks
  if (cur_worker < a) {
    start_task = cur_worker * n1;
    stop_task = (cur_worker + 1) * n1 - 1;
  } else {
    start_task = (a * n1) + (cur_worker - a) * n2;
    stop_task = (a * n1) + (cur_worker - a + 1) * n2 - 1;
  }
  return true;
}

} // namespace split
