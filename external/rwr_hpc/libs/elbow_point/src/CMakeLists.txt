# Gather source files for current normalization method(s)
set(ELBOW_SRC
  elbow_point.cpp
)

# OpenMP support
if(USE_OPENMP)
  find_package(OpenMP QUIET)
  if(OpenMP_FOUND)
    message(STATUS "OpenMP enabled for elbow_point library")
    add_definitions(-DUSE_OPENMP=1)
    set(USE_OPENMP_FLAG ON)
  else()
    message(WARNING "OpenMP requested but not found")
  endif()
endif()

# Define the correlate library target
add_library(elbow_point ${ELBOW_SRC})

target_include_directories(elbow_point
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
)

# Link other libraries
target_link_libraries(elbow_point
  PRIVATE
    utils)

# Link OpenMP if enabled
if(USE_OPENMP_FLAG)
  target_link_libraries(elbow_point PRIVATE OpenMP::OpenMP_CXX)
endif()
