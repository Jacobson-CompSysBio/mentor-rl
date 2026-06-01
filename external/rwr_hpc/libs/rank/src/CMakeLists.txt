# Gather source files for current normalization method(s)
set(RANK_SRC
  rank.cpp
)

# OpenMP support
if(USE_OPENMP)
  find_package(OpenMP QUIET)
  if(OpenMP_FOUND)
    message(STATUS "OpenMP enabled for rank library")
    add_definitions(-DUSE_OPENMP=1)
    set(USE_OPENMP_FLAG ON)
  else()
    message(WARNING "OpenMP requested but not found")
  endif()
endif()

# Define the rank library target
add_library(rank ${RANK_SRC})

target_include_directories(rank
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link OpenMP if enabled
if(USE_OPENMP_FLAG)
  target_link_libraries(rank PUBLIC OpenMP::OpenMP_CXX)
endif()

