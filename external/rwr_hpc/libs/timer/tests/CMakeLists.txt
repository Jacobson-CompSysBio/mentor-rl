add_executable(test_timer test_timer.cpp)

target_link_libraries(test_timer
  PRIVATE
    GTest::gtest_main
    GTest::gmock
    timer
)

include(GoogleTest)
gtest_discover_tests(test_timer)
