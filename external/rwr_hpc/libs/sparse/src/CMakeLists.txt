# Gather source files for current sparse method(s)
set(SPARSE_SRC
  CSR_Matrix.cpp
)

# Define the sparse library target
add_library(sparse ${SPARSE_SRC})

target_include_directories(sparse
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
)