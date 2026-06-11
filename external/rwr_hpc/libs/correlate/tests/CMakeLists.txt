# Find GoogleTest (either FetchContent or package)
# find_package(GTest REQUIRED)
# Fetch GoogleTest (portable). If you prefer a system package, replace this
# block with:  find_package(GTest REQUIRED)  and remove FetchContent bits.
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
# For MSVC runtime consistency with parent
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# --------- Enable HIP if needed ---------
if(USE_HIP)
  enable_language(HIP)
  message("Adding HIP libraries to tests")
  find_package(HIP REQUIRED) 
  find_package(hipblas REQUIRED)
endif()

# --------- TEST EXECUTABLES ---------
# Dot Product
add_executable(test_dot_local_math
  ${CMAKE_CURRENT_SOURCE_DIR}/test_dot_local_math.cpp
)
target_sources(test_dot_local_math PRIVATE $<TARGET_OBJECTS:dot_math_obj>)
target_include_directories(test_dot_local_math PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_link_libraries(test_dot_local_math PRIVATE GTest::gtest GTest::gtest_main GTest::gmock)
if(USE_HIP)
  target_link_libraries(test_dot_local_math PRIVATE hip::device hipblas)
  set_target_properties(test_dot_local_math PROPERTIES LINKER_LANGUAGE HIP)
  target_compile_definitions(test_dot_local_math PRIVATE USE_HIP=1)
endif()

add_executable(test_dot_local_dispatch
  ${CMAKE_CURRENT_SOURCE_DIR}/test_dot_local_dispatch.hip
)
target_link_libraries(test_dot_local_dispatch PRIVATE GTest::gtest GTest::gtest_main correlate)
target_include_directories(test_dot_local_dispatch PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
if(USE_HIP)
  set_target_properties(test_dot_local_dispatch PROPERTIES LINKER_LANGUAGE HIP)
  target_compile_definitions(test_dot_local_dispatch PRIVATE USE_HIP=1)
endif()

# Pearson
add_executable(test_pearson_local_math
  ${CMAKE_CURRENT_SOURCE_DIR}/test_pearson_local_math.cpp
)
target_sources(test_pearson_local_math PRIVATE $<TARGET_OBJECTS:pearson_math_obj>)
target_include_directories(test_pearson_local_math PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_link_libraries(test_pearson_local_math PRIVATE GTest::gtest GTest::gtest_main GTest::gmock normalize)
if(USE_HIP)
  target_link_libraries(test_pearson_local_math PRIVATE hip::device hipblas)
  set_target_properties(test_pearson_local_math PROPERTIES LINKER_LANGUAGE HIP)
  target_compile_definitions(test_pearson_local_math PRIVATE USE_HIP=1)
endif()

add_executable(test_pearson_local_dispatch
  ${CMAKE_CURRENT_SOURCE_DIR}/test_pearson_local_dispatch.hip
)
target_link_libraries(test_pearson_local_dispatch PRIVATE GTest::gtest GTest::gtest_main correlate)
target_include_directories(test_pearson_local_dispatch PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
if(USE_HIP)
  set_target_properties(test_pearson_local_dispatch PROPERTIES LINKER_LANGUAGE HIP)
  target_compile_definitions(test_pearson_local_dispatch PRIVATE USE_HIP=1)
endif()

# Spearman
add_executable(test_spearman_local_math
  ${CMAKE_CURRENT_SOURCE_DIR}/test_spearman_local_math.cpp
)
target_sources(test_spearman_local_math PRIVATE $<TARGET_OBJECTS:spearman_math_obj>)
target_include_directories(test_spearman_local_math PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_link_libraries(test_spearman_local_math PRIVATE GTest::gtest GTest::gtest_main GTest::gmock rank normalize)
if(USE_HIP)
  target_link_libraries(test_spearman_local_math PRIVATE hip::device hipblas)
  set_target_properties(test_spearman_local_math PROPERTIES LINKER_LANGUAGE HIP)
  target_compile_definitions(test_spearman_local_math PRIVATE USE_HIP=1)
endif()

add_executable(test_spearman_local_dispatch
  ${CMAKE_CURRENT_SOURCE_DIR}/test_spearman_local_dispatch.hip
)
target_link_libraries(test_spearman_local_dispatch PRIVATE GTest::gtest GTest::gtest_main correlate)
target_include_directories(test_spearman_local_dispatch PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
if(USE_HIP)
  set_target_properties(test_spearman_local_dispatch PROPERTIES LINKER_LANGUAGE HIP)
  target_compile_definitions(test_spearman_local_dispatch PRIVATE USE_HIP=1)
endif()

# Cosine
add_executable(test_cos_local_math
  ${CMAKE_CURRENT_SOURCE_DIR}/test_cos_local_math.cpp
)
target_sources(test_cos_local_math PRIVATE $<TARGET_OBJECTS:cosine_math_obj>)
target_include_directories(test_cos_local_math PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_link_libraries(test_cos_local_math PRIVATE GTest::gtest GTest::gtest_main GTest::gmock)

add_executable(test_cos_local_dispatch
  ${CMAKE_CURRENT_SOURCE_DIR}/test_cos_local_dispatch.hip
)
target_link_libraries(test_cos_local_dispatch PRIVATE GTest::gtest GTest::gtest_main correlate)
target_include_directories(test_cos_local_dispatch PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
if(USE_HIP)
  set_target_properties(test_cos_local_dispatch PROPERTIES LINKER_LANGUAGE HIP)
  target_compile_definitions(test_cos_local_dispatch PRIVATE USE_HIP=1)
endif()

# ---- Register with CTest ----
include(GoogleTest)
gtest_discover_tests(test_dot_local_dispatch)
gtest_discover_tests(test_dot_local_math)
gtest_discover_tests(test_pearson_local_dispatch)
gtest_discover_tests(test_pearson_local_math)
gtest_discover_tests(test_spearman_local_dispatch)
gtest_discover_tests(test_spearman_local_math)
gtest_discover_tests(test_cos_local_dispatch)
gtest_discover_tests(test_cos_local_math)
