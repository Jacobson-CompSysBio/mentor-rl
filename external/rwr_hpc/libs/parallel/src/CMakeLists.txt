# Gather source files for current parallel method(s)
set(PARALLEL_SRC
  parallel.cpp
)
# MPI Support
find_package(MPI REQUIRED)

# Define the library
add_library( parallel ${PARALLEL_SRC})

# Specify include directories for users of this library
target_include_directories(parallel
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link against dependencies:
target_link_libraries(
  parallel
  PUBLIC
  MPI::MPI_CXX    # library for MPI
  utils
)
