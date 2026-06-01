cmake_minimum_required(VERSION 3.15)
project(shortest_paths LANGUAGES CXX)

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

find_package(OpenMP REQUIRED)

# === Define Executable ===
add_executable(
  shortest_paths
  src/shortest_paths.cpp
)

# === Link CLI11 and MPI ===
target_link_libraries(
  shortest_paths      # target
  PRIVATE             # sets visibility to internal only 
  CLI11::CLI11        # library for CLI parsing
  OpenMP::OpenMP_CXX  # library for OpenMP
  multiplex
  timer
  file_io
)
