/// @file hip_common.hpp
/// @brief HIP functions and macros available to any normalization technique.
///
/// @author Ken Smith
/// @date 2025-07-24

#pragma once
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

/// @namespace normalize
/// @brief Namespace for column-wise standardization routines for dense data.
namespace normalize {

/// @brief Macro to check the return status of a HIP runtime API call.
///
/// If the HIP call fails (i.e., returns something other than hipSuccess),
/// this macro prints an error message to stderr including the file and line number,
/// then terminates the program with exit code 1.
///
/// @param func The HIP runtime function call to evaluate (e.g., hipMalloc, hipMemcpy).
///
/// @note This macro is intended for debugging and safety. It should wrap only HIP
///       API calls that return `hipError_t`.
///
/// @example
/// CHECK_HIP(hipMalloc(&ptr, size));
#define CHECK_HIP(func)                                   \
{                                                         \
  hipError_t status = (func);                             \
  if (status != hipSuccess) {                             \
    std::fprintf(stderr, "HIP API failed at %s:%d: %s\n", \
      __FILE__, __LINE__, hipGetErrorString(status));     \
    std::exit(EXIT_FAILURE);                              \
  }                                                       \
}                                                         \

bool hip_available();

} // namespace normalize
