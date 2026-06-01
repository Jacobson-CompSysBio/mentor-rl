# Gather source files for current normalization method(s)
set(SAMPLING_SRC
  sampling.cpp
)

# Define the rank library target
add_library(sampling ${SAMPLING_SRC})

target_include_directories(sampling
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
)

