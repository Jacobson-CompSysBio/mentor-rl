cmake_minimum_required(VERSION 3.15)
project(clean_edge_list LANGUAGES CXX)

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

# === Define Executable ===
add_executable(
  clean_edge_list
  src/clean_edge_list.cpp
)

# === Link CLI11 and MPI ===
target_link_libraries(
  clean_edge_list      # target
  PRIVATE             # sets visibility to internal only 
  CLI11::CLI11        # library for CLI parsing
)
