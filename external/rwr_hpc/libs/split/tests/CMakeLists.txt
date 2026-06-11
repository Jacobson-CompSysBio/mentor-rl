add_executable(test_split test_split.cpp)

target_link_libraries(test_split
  PRIVATE
    split
    GTest::gtest_main
    GTest::gmock
)

include(GoogleTest)
gtest_discover_tests(test_split)

