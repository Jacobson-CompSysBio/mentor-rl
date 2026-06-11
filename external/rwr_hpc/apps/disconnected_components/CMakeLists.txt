# apps/disconnected_components/CMakeLists.txt

add_executable(disconnected_components
  disconnected_components.cpp
)

# Require C++20 (change if needed)
target_compile_features(disconnected_components PRIVATE cxx_std_20)

# === Fetch CLI11 ===
FetchContent_Declare(
  CLI11
  GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
  GIT_TAG        v2.3.2 # or latest stable version
)

FetchContent_MakeAvailable(CLI11)

# Link against your libraries
target_link_libraries(disconnected_components
  PRIVATE
    CLI11::CLI11
    multiplex
    network
)

# Optional: stricter warnings for app only
if (MSVC)
  target_compile_options(disconnected_components PRIVATE /W4)
else()
  target_compile_options(disconnected_components PRIVATE -Wall -Wextra -Wpedantic)
endif()
