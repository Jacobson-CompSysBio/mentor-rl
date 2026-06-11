add_executable(test_parallel_internal test_parallel_internal.cpp)
# add_executable(test_parallel test_parallel.cpp)

# MPI Support
find_package(MPI REQUIRED)

target_include_directories(test_parallel_internal
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
)

target_link_libraries(test_parallel_internal
  PRIVATE
    GTest::gtest_main
    GTest::gmock
    MPI::MPI_CXX        # library for MPI
)

# target_link_libraries(test_parallel
#   PRIVATE
#     GTest::gtest_main
#     GTest::gmock
#     MPI::MPI_CXX        # library for MPI
#     parallel
#     /usr/lib64/libxpmem.so                              # library for GPU-aware MPI
#     /opt/cray/pe/mpich/8.1.30/gtl/lib/libmpi_gtl_hsa.so # library for GPU-aware MPI
# )

include(GoogleTest)
gtest_discover_tests(test_parallel_internal)
# gtest_discover_tests(test_parallel)
