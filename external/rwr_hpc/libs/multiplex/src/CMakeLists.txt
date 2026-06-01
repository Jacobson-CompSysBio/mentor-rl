# Gather source files for network class
set(MULTIPLEX_SRC
  Multiplex.cpp
)

# OpenMP support
if(USE_OPENMP)
  find_package(OpenMP QUIET)
  if(OpenMP_FOUND)
    message(STATUS "OpenMP enabled for multiplex library")
    add_definitions(-DUSE_OPENMP=1)
    set(USE_OPENMP_FLAG ON)
  else()
    message(WARNING "OpenMP requested but not found")
  endif()
endif()

# Define the Netowrk library target
add_library(multiplex ${MULTIPLEX_SRC})

# Specify include directories for users of this library
target_include_directories(multiplex
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
)

# Link main libraries
target_link_libraries(multiplex
  PUBLIC    # Allow programs that link to multiplex to also link to network and sparse
    network # library for networks
    sparse  # library for sparse matrix
)

# Link OpenMP if enabled
if(USE_OPENMP_FLAG)
  target_link_libraries(multiplex
    PRIVATE
      OpenMP::OpenMP_CXX # library for OpenMP
  )
endif()