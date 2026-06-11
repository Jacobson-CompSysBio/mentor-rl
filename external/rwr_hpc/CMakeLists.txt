cmake_minimum_required(VERSION 3.15)
project(jailtools LANGUAGES CXX)

# Specify C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

option(BUILD_TESTS "Enable unit tests" OFF)

if(BUILD_TESTS)
  # === GoogleTest setup ===
  include(FetchContent)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/release-1.12.1.zip
  )
  FetchContent_MakeAvailable(googletest)
  enable_testing()
endif()

# Add libraries
add_subdirectory(libs/sparse)
add_subdirectory(libs/network)
add_subdirectory(libs/multiplex)
add_subdirectory(libs/timer)
add_subdirectory(libs/sort)
add_subdirectory(libs/file_io)

# Add apps
add_subdirectory(apps/shortest_paths)
add_subdirectory(apps/clean_edge_list)
add_subdirectory(apps/gene_layer_map)
add_subdirectory(apps/disconnected_components)
