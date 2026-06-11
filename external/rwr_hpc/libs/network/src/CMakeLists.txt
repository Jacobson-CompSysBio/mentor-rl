# Gather source files for network class
set(NETWORK_SRC
  Network.cpp
)

# OpenMP support
if(USE_OPENMP)
  find_package(OpenMP QUIET)
  if(OpenMP_FOUND)
    message(STATUS "OpenMP enabled for network library")
    add_definitions(-DUSE_OPENMP=1)
    set(USE_OPENMP_FLAG ON)
  else()
    message(WARNING "OpenMP requested but not found")
  endif()
endif()

# Define the Netowrk library target
add_library(network ${NETWORK_SRC})

# Specify include directories for users of this library
target_include_directories(network
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
)

# Link main libraries
target_link_libraries(network
  PUBLIC
    sort               # library for sorting 
    sparse             # library for sparse matrix
)

# Link OpenMP if enabled
if(USE_OPENMP_FLAG)
  target_link_libraries(network
    PRIVATE
      OpenMP::OpenMP_CXX # library for OpenMP
  )
endif()