add_executable(test_elbow_point test_elbow_point.cpp)

target_link_libraries(test_elbow_point
  PRIVATE
    elbow_point
    GTest::gtest_main
    GTest::gmock
)

include(GoogleTest)
gtest_discover_tests(test_elbow_point)

