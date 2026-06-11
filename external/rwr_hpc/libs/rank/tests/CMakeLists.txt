add_executable(test_rank test_rank.cpp ../src/rank.cpp)

target_link_libraries(test_rank
  PRIVATE
    GTest::gtest_main
    GTest::gmock
)

include(GoogleTest)
gtest_discover_tests(test_rank)
