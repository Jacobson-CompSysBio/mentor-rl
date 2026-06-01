# Gather source files for current rwr method(s)
set(RWR_SRC
  utils.cpp
  RWR_Worker.hip
)

# === Enable ROCm-aware MPI if applicable
find_package(MPI REQUIRED)
find_package(HIP REQUIRED)
find_package(hipblas REQUIRED)
find_package(hipsparse REQUIRED)

# Define the correlate library target
add_library(rwr ${RWR_SRC})

# Specify include directories for users of this library
target_include_directories(rwr
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link other libraries
target_link_libraries(
  rwr
  PRIVATE
  MPI::MPI_CXX
  hip::device     # library for device (GPU) functions and flags
  hip::host       # library for host (CPU) functions and linking to other libraries
  hipblas         # library for hipBLAS functions
  hipsparse       # library for hipSPARSE functions
  utils
  sampling
  sparse
  batch_generator
  split
)

# Optional: Additional HIP flags
set_target_properties(rwr PROPERTIES
  HIP_ARCHITECTURES "gfx90a"
)
