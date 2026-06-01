#include "elbow_point/elbow_point.hpp"

#ifdef USE_OPENMP
#include <omp.h>
#endif

#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <utils/vector_utils.hpp>

namespace elbow_point {

std::pair<double, double> elbow_point(const std::vector<double>& x, const std::vector<double>& y) {
  // Check length of vectors
  if (x.size() != y.size()) {
    throw std::invalid_argument("elbow_point - x and y must be the same size");
  }
  if (x.size() < 2) {
    throw std::invalid_argument("elbow_point - x and y must have at least 2 points");
  }

  // // Check for duplicates in x
  // if (utils::has_duplicates(x)) {
  //   throw std::invalid_argument("elbow_point - x has duplicate values");
  // }

  // start at the end point poistion
  std::size_t idx_of_min_x = utils::idx_of_min_element(x);
  std::size_t idx_of_max_x = utils::idx_of_max_element(x);

  // start point
  double x_start = x[idx_of_min_x];
  double y_start = y[idx_of_min_x];
  
  // end point
  double x_end = x[idx_of_max_x];
  double y_end = y[idx_of_max_x];

  // distance between x,y start and end
  double dx = x_end - x_start;
  double dy = y_end - y_start;
  double d_start_end = std::sqrt(dx*dx + dy*dy);

  // distance between each x-y pair and the line passing through the end points
  std::vector<double> d(x.size());
  #ifdef USE_OPENMP
  #pragma omp parallel for schedule(static)
  #endif
  for (std::size_t i = 0; i < x.size(); ++i) {
    d[i] = std::abs(dx * (y_start - y[i]) - dy * (x_start - x[i]));
  }

  std::size_t idx_if_max_d = utils::idx_of_max_element(d);
  return std::make_pair(x[idx_if_max_d], y[idx_if_max_d]);
}

} // namespace  elbow_point
