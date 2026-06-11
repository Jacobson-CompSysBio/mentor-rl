/// @file BatchGen.hpp
/// @brief Header-only utility for generating contiguous index batches over a range.
///
/// Useful for partitioning work (e.g., iteration ranges, task indices) into batches
/// of fixed size for parallel or staged processing.
///
/// @author Ken Smith
/// @date 2025-07-24

#include <utility>      // for std::pair
#include <stdexcept>    // for std::invalid_argument
#include <type_traits>  // for std::is_arithmetic
#include <algorithm>    // for std::min

/// @class BatchGen
/// @brief A generic batch index generator for iterating over a numeric range in fixed-size blocks.
///
/// This class produces successive ranges of indices from `[min, max]`, split into batches
/// of user-specified size. Each call to `next()` returns a `(start, end)` pair (inclusive),
/// and `done()` indicates when all batches have been exhausted.
///
/// @tparam T An arithmetic type (e.g., int, long, std::size_t). Must support comparison and arithmetic operators.
///
/// @throws std::invalid_argument if `min > max`.
/// @throws invalid_argument if `batch_size <= 0`.
///
/// @example
/// BatchGen<int> gen(0, 99, 10);
/// while (!gen.done()) {
///   auto [start, end] = gen.next();
///   // process elements from start to end
/// }
template<typename T>
class BatchGen {
  static_assert(std::is_arithmetic<T>::value, "BatchGen requires an arithmetic type.");

  public:
  /// @brief Constructs a BatchGen object with the specified range and batch size.
  ///
  /// @param min The start of the range (inclusive).
  /// @param max The end of the range (inclusive).
  /// @param batch_size The maximum size of each batch.
  ///
  /// @throws std::invalid_argument if `min > max`
  /// @throws std::invalid_argument if `batch_size <= 0`.
  BatchGen(const T min, const T max, const T batch_size)
  : current(min), min_val(min), max_val(max), batch_size(batch_size) {
    static_assert(std::is_integral<T>::value, "T must be an integral type");

    if (min > max || batch_size <= 0) {
      throw std::invalid_argument("BatchGen - Invalid range or batch size.");
    }
  }

  /// @brief Returns the `[start, end]` indices of the current batch and advances to the next.
  ///
  /// @return A pair of indices [start, end] (inclusive). If done, returns a sentinel pair `{max + 1, max + 1}`.
  std::pair<T, T> next() {
    if (done()) return {max_val + 1, max_val + 1};  // Sentinel

    T start = current;
    T end = std::min(current + batch_size - 1, max_val);
    current = end + static_cast<T>(1);
    return {start, end};
  }

  /// @brief Checks whether all batches have been processed.
  ///
  /// @return `true` if no more batches remain; `false` otherwise.
  bool done() const {
    return current > max_val;
  }

  /// @brief Resets the generator to the beginning of the range.
  void reset() {
    current = min_val;
  }

private:
  T current;     ///< Current index position
  T min_val;     ///< Minimum range value (inclusive)
  T max_val;     ///< Maximum range value (inclusive)
  T batch_size;  ///< Batch size
};
