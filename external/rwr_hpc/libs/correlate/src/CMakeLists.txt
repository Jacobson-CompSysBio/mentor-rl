# --- Enable HIP if needed ---
if(USE_HIP)
  enable_language(HIP)
endif()

# --------- OBJECT LIBRARIES FOR SHARED MATH CODE ---------
# Dot math internals
set(DOT_MATH_OBJECT_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/metrics/dot/dot_cpu.cpp
)
if(USE_HIP)
  list(APPEND DOT_MATH_OBJECT_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/common/hip_common.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/metrics/dot/dot_gpu.hip
  )
endif()
add_library(dot_math_obj OBJECT ${DOT_MATH_OBJECT_SOURCES})

# Pearson math internals
set(PEARSON_MATH_OBJECT_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/metrics/pearson/pearson_cpu.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/metrics/dot/dot_cpu.cpp
)
if(USE_HIP)
  list(APPEND PEARSON_MATH_OBJECT_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/common/hip_common.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/metrics/pearson/pearson_gpu.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/metrics/dot/dot_gpu.hip
  )
endif()
add_library(pearson_math_obj OBJECT ${PEARSON_MATH_OBJECT_SOURCES})
target_include_directories(pearson_math_obj
  PUBLIC
    $<TARGET_PROPERTY:normalize,INTERFACE_INCLUDE_DIRECTORIES>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Spearman math internals
set(SPEARMAN_MATH_OBJECT_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/metrics/spearman/spearman_cpu.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/metrics/pearson/pearson_cpu.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/metrics/dot/dot_cpu.cpp
)
if(USE_HIP)
  list(APPEND SPEARMAN_MATH_OBJECT_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/common/hip_common.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/metrics/spearman/spearman_gpu.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/metrics/pearson/pearson_gpu.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/metrics/dot/dot_gpu.hip
  )
endif()
add_library(spearman_math_obj OBJECT ${SPEARMAN_MATH_OBJECT_SOURCES})

target_include_directories(spearman_math_obj
  PUBLIC
    $<TARGET_PROPERTY:rank,INTERFACE_INCLUDE_DIRECTORIES>
    $<TARGET_PROPERTY:normalize,INTERFACE_INCLUDE_DIRECTORIES>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)


# Cosine math internals
set(COSINE_MATH_OBJECT_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/metrics/cos/cos_cpu.cpp
)
add_library(cosine_math_obj OBJECT ${COSINE_MATH_OBJECT_SOURCES})
target_include_directories(cosine_math_obj
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# --------- MAIN LIBRARY ---------
add_library(correlate)

target_sources(correlate
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/correlate/correlate.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/correlate/dot.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/correlate/pearson.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/correlate/spearman.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/correlate/cos.hpp
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/metrics/dot/dot_dispatch.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metrics/pearson/pearson_dispatch.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metrics/spearman/spearman_dispatch.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metrics/cos/cos_dispatch.cpp
    $<TARGET_OBJECTS:dot_math_obj>
    $<TARGET_OBJECTS:pearson_math_obj>
    $<TARGET_OBJECTS:spearman_math_obj>
    $<TARGET_OBJECTS:cosine_math_obj>
)

target_include_directories(correlate
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(correlate PROPERTIES POSITION_INDEPENDENT_CODE ON)
if (MSVC)
  target_compile_options(correlate PRIVATE /W4 /permissive-)
else()
  target_compile_options(correlate PRIVATE -Wall -Wextra -Wpedantic)
endif()

# ---- OpenMP (conditional) ------
if(USE_OPENMP)
  find_package(OpenMP QUIET)
  if(OpenMP_FOUND)
    target_compile_definitions(correlate PRIVATE USE_OPENMP=1)
    message(STATUS "OpenMP enabled for correlate library")
  else()
    message(WARNING "OpenMP requested but not found for correlate library")
  endif()
endif()

# ---- HIP/ROCm (conditional) ----
if (USE_HIP)
  find_package(HIP REQUIRED)
  message(STATUS "HIP enabled for correlate library")

  find_package(hipblas QUIET)
  if (NOT hipblas_FOUND)
    find_package(hipBLAS QUIET)
  endif()
  if (hipblas_FOUND)
    set(_HIPBLAS_TARGET hipblas)
  elseif (hipBLAS_FOUND)
    set(_HIPBLAS_TARGET hipBLAS)
  else()
    message(FATAL_ERROR "hipBLAS not found. Install ROCm hipBLAS or set HIPBLAS_DIR.")
  endif()
  target_compile_definitions(correlate PRIVATE USE_HIP=1)
  target_sources(correlate PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/common/hip_common.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/common/hip_seams.cpp
  )
endif()

# ---- MPI (conditional) ----
if (USE_MPI)
  find_package(MPI)
  message(STATUS "MPI enabled for correlate library")
  target_compile_definitions(correlate PRIVATE USE_MPI=1)
endif()

set(_CORRELATE_PUBLIC_DEPS rank normalize)
if(USE_OPENMP)
  list(APPEND _CORRELATE_PUBLIC_DEPS OpenMP::OpenMP_CXX)
endif()
if(USE_HIP)
  list(APPEND _CORRELATE_PUBLIC_DEPS hip::device ${_HIPBLAS_TARGET})
endif()
if(USE_MPI)
  list(APPEND _CORRELATE_PUBLIC_DEPS MPI::MPI_CXX)
endif()
target_link_libraries(correlate PUBLIC ${_CORRELATE_PUBLIC_DEPS})
