add_executable(test_vector_utils
  test_vector_utils.cpp)

target_link_libraries(test_vector_utils
  PRIVATE
    GTest::gtest_main
    GTest::gmock
    utils
)

include(GoogleTest)
gtest_discover_tests(test_vector_utils)
