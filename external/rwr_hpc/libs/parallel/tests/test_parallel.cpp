#include <gtest/gtest.h>

// #ifdef USE_MPI
#include <parallel/parallel.hpp>

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  MPI_Init(&argc, &argv);
  int result = RUN_ALL_TESTS();
  MPI_Finalize();
  return result;
}

TEST(Parallel, GetCommRank) {
  int ref_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &ref_rank);
  EXPECT_EQ(parallel::get_comm_rank(MPI_COMM_WORLD), ref_rank);

}

TEST(Parallel, GetCommSize) {
  int ref_size;
  MPI_Comm_size(MPI_COMM_WORLD, &ref_size);
  EXPECT_EQ(parallel::get_comm_size(MPI_COMM_WORLD), ref_size);
}

TEST(Parallel, GatherN_Elements) {
  int comm_size = parallel::get_comm_size();
  int comm_rank = parallel::get_comm_rank();

  std::size_t n_elements = static_cast<std::size_t>(comm_rank + 1);
  auto elements = parallel::gather_n_elements(n_elements, comm_size, 0);

  if (comm_rank == 0) {
    ASSERT_EQ(elements.size(), comm_size);
    for (std::size_t i = 0; i < comm_size; ++i) {
      EXPECT_EQ(elements[i], i+1);
    }
  } else {
    EXPECT_EQ(elements.size(), 0);
  }
}

TEST(Parallel, BcastN_Elements) {
  int comm_size = parallel::get_comm_size();
  std::vector<std::size_t> n_elements(comm_size);

  if (parallel::get_comm_rank() == 0) {
    for (std::size_t i = 0; i < comm_size; ++i) {
      n_elements[i] = i + 2;
    }
  }

  parallel::bcast_n_elements(n_elements, 0);

  ASSERT_EQ(n_elements.size(), comm_size);
    for (std::size_t i = 0; i < comm_size; ++i) {
    EXPECT_EQ(n_elements[i], i+2);
  }
}

TEST(Parallel, GatherWithGatherV) {
  int comm_rank = parallel::get_comm_rank();
  int comm_size = parallel::get_comm_size();
  std::size_t vec_size = 10;

  std::vector<double> local_data(vec_size, static_cast<double>(comm_rank + 1));
  std::vector<double> out;

  parallel::gather(out, local_data, 0);

  if (comm_rank == 0) {
    ASSERT_EQ(out.size(), vec_size * comm_size);

    for (int i = 0; i < comm_size; ++i) {
      for (std::size_t j = 0; j < vec_size; ++j) {
        EXPECT_DOUBLE_EQ(out[i * vec_size + j], static_cast<double>(i + 1));
      }
    }
  } else {
    EXPECT_EQ(out.size(), 0);
  }
}

TEST(Parallel, GatherWithRecv) {
  int comm_rank = parallel::get_comm_rank();
  int comm_size = parallel::get_comm_size();
  std::size_t vec_size = static_cast<std::size_t>(std::numeric_limits<int>::max() / comm_size + 1);

  std::vector<double> local_data(vec_size, static_cast<double>(comm_rank + 1));
  std::vector<double> out;

  parallel::gather(out, local_data, 0);

  if (comm_rank == 0) {
    ASSERT_EQ(out.size(), vec_size * comm_size);

    for (int i = 0; i < comm_size; ++i) {
      for (std::size_t j = 0; j < vec_size; ++j) {
        EXPECT_DOUBLE_EQ(out[i * vec_size + j], static_cast<double>(i + 1));
      }
    }
  } else {
    EXPECT_EQ(out.size(), 0);
  }
}

TEST(Parallel, GatherWithRecvBatched) {
  int comm_rank = parallel::get_comm_rank();
  int comm_size = parallel::get_comm_size();
  std::size_t vec_size = static_cast<std::size_t>(std::numeric_limits<int>::max()) + 1UL;

  std::vector<double> local_data(vec_size, static_cast<double>(comm_rank + 1));
  std::vector<double> out;

  parallel::gather(out, local_data, 0);

  if (comm_rank == 0) {
    ASSERT_EQ(out.size(), vec_size * comm_size);

    for (int i = 0; i < comm_size; ++i) {
      for (std::size_t j = 0; j < vec_size; ++j) {
        EXPECT_DOUBLE_EQ(out[i * vec_size + j], static_cast<double>(i + 1));
      }
    }
  } else {
    EXPECT_EQ(out.size(), 0);
  }
}

TEST(Parallel, GatherAndRowJoinColMajor) {
  std::vector<double> out;

  std::vector<double> local_in(10);
  const std::size_t n_cols = 5;

  int comm_rank = parallel::get_comm_rank();
  int comm_size = parallel::get_comm_size();
  for (std::size_t i = 0; i < 10; ++i) {
    local_in[i] = 10 * comm_rank + i;
  }

  parallel::gather_and_row_join_column_major(out, local_in, n_cols);

  if (comm_rank == 0) {
    ASSERT_EQ(out.size(), 10 * comm_size);

  } else {
    EXPECT_EQ(out.size(), 0);
  }
}

// #endif