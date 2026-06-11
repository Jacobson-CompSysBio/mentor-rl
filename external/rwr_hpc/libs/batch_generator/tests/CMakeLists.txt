add_executable(test_batch_generator test_batch_generator.cpp)

target_link_libraries(test_batch_generator
  PRIVATE
    batch_generator
    GTest::gtest_main
    GTest::gmock
)

include(GoogleTest)
gtest_discover_tests(test_batch_generator)
