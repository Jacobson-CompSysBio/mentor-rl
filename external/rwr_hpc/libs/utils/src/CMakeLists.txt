# Gather source files for current vector utils methods
set(VECTOR_UTILS_SRC
  vector_utils.cpp
)

# Define the vector utils library target
add_library(utils ${VECTOR_UTILS_SRC})

target_include_directories(utils
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
)