add_executable(test_multiplex test_multiplex.cpp)

target_link_libraries(test_multiplex
  PRIVATE
    GTest::gtest_main
    GTest::gmock
    multiplex
    network
    sparse
    utils
)

include(GoogleTest)
gtest_discover_tests(test_multiplex)
