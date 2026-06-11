add_executable(test_network test_network.cpp)

target_link_libraries(test_network
  PRIVATE
    network
    GTest::gtest_main
    GTest::gmock
)

include(GoogleTest)
gtest_discover_tests(test_network)
