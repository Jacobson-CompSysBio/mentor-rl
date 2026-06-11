add_executable(test_csr_matrix test_csr_matrix.cpp)

target_link_libraries(test_csr_matrix
  PRIVATE
    sparse
    GTest::gtest_main
    GTest::gmock
)

include(GoogleTest)
gtest_discover_tests(test_csr_matrix)
