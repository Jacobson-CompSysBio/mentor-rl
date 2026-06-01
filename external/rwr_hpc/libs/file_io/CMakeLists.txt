cmake_minimum_required(VERSION 3.15)
project(file_io LANGUAGES CXX)

# Specify C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Define the library
add_library(
  file_io
  src/matrix_io.cpp
  src/binary_io.cpp
)

# Specify include directories for users of this library
target_include_directories(
  file_io
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Optionally include tests
if(BUILD_TESTS)
  add_subdirectory(tests)
endif()