add_executable(test_binary_io test_binary_io.cpp)
add_executable(test_matrix_io test_matrix_io.cpp)
add_executable(test_vector_io test_vector_io.cpp)

target_link_libraries(test_binary_io
  PRIVATE
    file_io
    GTest::gtest_main
    GTest::gmock
)

target_link_libraries(test_matrix_io
  PRIVATE
    file_io
    GTest::gtest_main
    GTest::gmock
)

target_link_libraries(test_vector_io
  PRIVATE
    file_io
    GTest::gtest_main
    GTest::gmock
)

include(GoogleTest)
gtest_discover_tests(test_binary_io)
gtest_discover_tests(test_matrix_io)
gtest_discover_tests(test_vector_io)
