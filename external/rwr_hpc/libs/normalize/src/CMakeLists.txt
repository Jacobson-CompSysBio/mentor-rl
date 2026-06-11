# --------- OBJECT LIBRARIES FOR STANDARD MATH CORE ---------
set(STANDARD_MATH_OBJECT_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/standard/standard_cpu.cpp
)
if(USE_HIP)
  list(APPEND STANDARD_MATH_OBJECT_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/common/hip_common.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/standard/standard_gpu.hip  
  )
endif()
add_library(standard_math_obj ${STANDARD_MATH_OBJECT_SOURCES})

add_library(normalize)
target_sources(normalize
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/normalize/standard.hpp
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/standard/standard_dispatch.cpp
    $<TARGET_OBJECTS:standard_math_obj>
)
    
target_include_directories(normalize
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(normalize PROPERTIES POSITION_INDEPENDENT_CODE ON)
if (MSVC)
  target_compile_options(normalize PRIVATE /W4 /permissive-)
else()
  target_compile_options(normalize PRIVATE -Wall -Wextra -Wpedantic)
endif()

# ---- OpenMP (conditional) ------
if(USE_OPENMP)
  find_package(OpenMP QUIET)
  if(OpenMP_FOUND)
    message(STATUS "OpenMP enabled for normalize library")
    add_definitions(-DUSE_OPENMP=1)
  else()
    message(WARNING "OpenMP requested but not found for normalize library")
  endif()
endif()

# ---- HIP/ROCm (conditional) ----
if(USE_HIP)
  find_package(HIP QUIET)
  if(HIP_FOUND)
    enable_language(HIP)
    message(STATUS "HIP support enabled for normalize library")
    target_compile_definitions(normalize PRIVATE USE_HIP=1)
    target_sources(normalize PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/common/hip_common.hip
      ${CMAKE_CURRENT_SOURCE_DIR}/common/hip_seams.cpp
    )
  else()
    message(WARNING "HIP not found. Building without GPU support.")
  endif()
endif()

set(_NORMALIZE_PUBLIC_DEPS)
# Link OpenMP if enabled
if(USE_OPENMP)
  list(APPEND PUBLIC OpenMP::OpenMP_CXX)
endif()
# Link HIP if present
if(USE_HIP)
  list(APPEND PUBLIC hip::device)
endif()
target_link_libraries(normalize PUBLIC ${_NORMALIZE_PUBLIC_DEPS})
