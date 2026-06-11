# Gather source files for current split method(s)
set(SPLIT_SRC
  split.cpp
)

# Define the split library target
add_library(split ${SPLIT_SRC})

target_include_directories(split
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
)