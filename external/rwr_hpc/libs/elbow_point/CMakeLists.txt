cmake_minimum_required(VERSION 3.15)
project(elbow_point LANGUAGES CXX)

# Specify C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

option(USE_OPENMP "Enable OpenMP in CPU backend" ON)

add_subdirectory(src)

# Optionally include tests
if(BUILD_TESTS)
  add_subdirectory(tests)
endif()