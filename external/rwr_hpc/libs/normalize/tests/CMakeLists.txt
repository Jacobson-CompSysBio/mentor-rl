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

if(USE_HIP)
  enable_language(HIP)
endif()

add_executable(test_standard_math
  ${CMAKE_CURRENT_SOURCE_DIR}/test_standard_math.cpp
)
target_sources(test_standard_math PRIVATE $<TARGET_OBJECTS:standard_math_obj>)
target_include_directories(test_standard_math PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_link_libraries(test_standard_math PRIVATE GTest::gtest GTest::gtest_main GTest::gmock)
if(USE_HIP)
  set_target_properties(test_standard_math PROPERTIES LINKER_LANGUAGE HIP)
  target_compile_definitions(test_standard_math PRIVATE USE_HIP=1)
endif()

add_executable(test_standard_dispatch
  ${CMAKE_CURRENT_SOURCE_DIR}/test_standard_dispatch.hip
)
target_link_libraries(test_standard_dispatch PRIVATE GTest::gtest GTest::gtest_main normalize)
target_include_directories(test_standard_dispatch PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
if(USE_HIP)
  set_target_properties(test_standard_dispatch PROPERTIES LINKER_LANGUAGE HIP)
  target_compile_definitions(test_standard_dispatch PRIVATE USE_HIP=1)
endif()

include(GoogleTest)
gtest_discover_tests(test_standard_dispatch)
gtest_discover_tests(test_standard_math)
