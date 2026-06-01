cmake_minimum_required(VERSION 3.15)
project(stats LANGUAGES CXX)

# Specify C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Declare header-only library as INTERFACE
add_library(stats INTERFACE)

# Specify include directories for users of this library
target_include_directories(
  stats
  INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Optionally include tests
if(BUILD_TESTS)
  add_subdirectory(tests)
endif()