add_executable(test_sampling test_sampling.cpp)

target_link_libraries(test_sampling
  PRIVATE
    GTest::gtest_main
    GTest::gmock
    sampling
)

include(GoogleTest)
gtest_discover_tests(test_sampling)
