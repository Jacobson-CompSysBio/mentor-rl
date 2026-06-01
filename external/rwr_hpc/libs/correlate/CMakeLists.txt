cmake_minimum_required(VERSION 3.15)
project(correlate VERSION 0.1.0 LANGUAGES CXX)

# Options
option(BUILD_TESTS "Build unit tests" ON)
option(USE_HIP     "Enable HIP/ROCm GPU support" ON)
option(USE_OPENMP  "Enable OpenMP support" ON)

# Global C++ settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Subdirectories
add_subdirectory(src)

if (BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
