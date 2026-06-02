cmake_minimum_required(VERSION 3.16)

project(mentor_runtime LANGUAGES CXX)

message(STATUS "DEBUG: using cpp_runtime/CMakeLists.txt from ${CMAKE_CURRENT_LIST_FILE}")

find_package(OpenMP)

option(
  MENTOR_USE_RWR_HPC
  "Build mentor_runtime with selected embedded RWR-HPC libraries"
  OFF
)

add_library(
  mentor_runtime
  SHARED
  src/c_api.cpp
  src/graph_store.cpp
  src/tool_kernels.cpp
  src/rwr_hpc_adapter.cpp
)

target_include_directories(
  mentor_runtime
  PUBLIC
    include
)

target_compile_features(mentor_runtime PUBLIC cxx_std_17)

if(OpenMP_CXX_FOUND)
  target_link_libraries(mentor_runtime PUBLIC OpenMP::OpenMP_CXX)
endif()

if(MENTOR_USE_RWR_HPC)
  message(FATAL_ERROR
    "MENTOR_USE_RWR_HPC is intentionally disabled for now. "
    "Use the standalone RWR-HPC app backend for full RWR++ workflows first."
  )
endif()

set_target_properties(
  mentor_runtime
  PROPERTIES
  OUTPUT_NAME mentor_runtime
  POSITION_INDEPENDENT_CODE ON
)