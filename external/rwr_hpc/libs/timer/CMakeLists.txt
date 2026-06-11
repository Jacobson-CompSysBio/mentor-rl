cmake_minimum_required(VERSION 3.15)
project(timer LANGUAGES CXX)

# Specify C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_subdirectory(src)

# Optionally include tests
if(BUILD_TESTS)
  add_subdirectory(tests)
endif()
