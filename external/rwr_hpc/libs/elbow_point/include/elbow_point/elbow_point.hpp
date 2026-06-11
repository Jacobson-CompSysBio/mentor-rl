/// @file elbow_point.hpp
/// @brief Compute the "elbow point" of a curve
/// 
/// Based on R/elbow_point.R by Joshua French and Mohammad Meysami
/// https://rdrr.io/cran/smerc/src/R/elbow_point.R
///
/// @author Ken Smith
/// @date 2025-07-24

#pragma once

#include <utility>
#include <vector>

/// @namespace utils
/// @brief Namespace for finding the elbow point of curves
namespace  elbow_point {

/// @brief Identifies the "elbow point" of a 2D curve using a geometric method.
///
/// Given a curve defined by vectors `x` and `y`, this function returns the point
/// with the greatest perpendicular distance from the line connecting the curve's
/// endpoints. This method is commonly used to detect the point of diminishing returns
/// (e.g., optimal `k` in k-means clustering, inflection in scree plots, etc.).
///
/// @param x The vector of x-coordinates. Must be strictly ordered (no duplicates).
/// @param y The vector of y-coordinates. Must be the same length as `x`.
/// @return A pair `(x, y)` representing the elbow point on the curve.
///
/// @throws std::invalid_argument if `x` and `y` are not the same size.
/// @throws std::invalid_argument if the input vectors contain fewer than 2 elements.
/// @throws std::invalid_argument if `x` contains duplicate values.
///
/// @note The function internally uses OpenMP to parallelize distance computation.
///
/// @see has_duplicates(), idx_of_max_element()
std::pair<double, double> elbow_point(const std::vector<double>& x, const std::vector<double>& y);

} // elbow_point  utils
