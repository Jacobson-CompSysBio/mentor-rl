cmake_minimum_required(VERSION 3.16)

project(mentor_runtime LANGUAGES CXX)

find_package(OpenMP)

add_library(
  mentor_runtime
  SHARED
  src/c_api.cpp
  src/graph_store.cpp
  src/tool_kernels.cpp
)

target_include_directories(mentor_runtime PUBLIC include)
target_compile_features(mentor_runtime PUBLIC cxx_std_17)
if(OpenMP_CXX_FOUND)
  target_link_libraries(mentor_runtime PUBLIC OpenMP::OpenMP_CXX)
endif()

set_target_properties(
  mentor_runtime
  PROPERTIES
  OUTPUT_NAME mentor_runtime
  POSITION_INDEPENDENT_CODE ON
)
