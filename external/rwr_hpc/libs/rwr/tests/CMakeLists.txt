add_executable(test_rwr_utils test_rwr_utils.hip)

target_link_libraries(test_rwr_utils
  PRIVATE
    GTest::gtest_main
    GTest::gmock
    rwr
    sparse
    sampling
    utils

)

# target_link_libraries(test_dot_product
#   PRIVATE
#     GTest::gtest_main
# )

include(GoogleTest)
gtest_discover_tests(test_rwr_utils)
# gtest_discover_tests(test_dot_product)
