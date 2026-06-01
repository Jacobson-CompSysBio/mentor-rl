cmake_minimum_required(VERSION 3.15)
project(rwr_ablation LANGUAGES CXX HIP)

# === Set C++ standard ===
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# === Enable FetchContent (built into CMake 3.11+) ===
include(FetchContent)

# === Fetch CLI11 ===
FetchContent_Declare(
    CLI11
    GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
    GIT_TAG        v2.3.2 # or latest stable version
)

FetchContent_MakeAvailable(CLI11)

# === Enable ROCm-aware MPI if applicable +++
find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)
find_package(HIP REQUIRED)
find_package(hipblas REQUIRED)
find_package(hipsparse REQUIRED)

# === Header Search Path ===
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# === Define Executable ===
add_executable(
  rwr_ablation
  src/main.hip
)

# === Link CLI11 and MPI ===
target_link_libraries(
  rwr_ablation        # target
  PRIVATE             # sets visibility to internal only 
  CLI11::CLI11        # library for CLI parsing
  OpenMP::OpenMP_CXX  # library for OpenMP
  MPI::MPI_CXX        # library for MPI
  hip::device         # library for device (GPU) functions and flags
  hip::host           # library for host (CPU) functions and linking to other libraries
  hipblas             # library for hipBLAS functions
  hipsparse           # library for hipSPARSE functions
  /usr/lib64/libxpmem.so                              # library for GPU-aware MPI
  /opt/cray/pe/mpich/8.1.30/gtl/lib/libmpi_gtl_hsa.so # library for GPU-aware MPI
  multiplex
  sparse
  parallel
  utils
  file_io
  correlate
  normalize
  rank
  rwr
  timer
)
