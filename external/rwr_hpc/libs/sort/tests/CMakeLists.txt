add_executable(test_sort test_sort.cpp)

target_link_libraries(test_sort
  PRIVATE
    sort
    GTest::gtest_main
    GTest::gmock
)

include(GoogleTest)
gtest_discover_tests(test_sort)
