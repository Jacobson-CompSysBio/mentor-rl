add_executable(test_stats test_stats.cpp)

target_link_libraries(test_stats
  PRIVATE
    GTest::gtest_main
    GTest::gmock
    stats
)

include(GoogleTest)
gtest_discover_tests(test_stats)
