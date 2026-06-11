#include <gtest/gtest.h>
#include <gmock/gmock-matchers.h>
#include "file_io/binary_io.hpp"
#include <vector>
#include <string>
#include <filesystem>

namespace fs = std::filesystem;

TEST(LoadBinaryBlock, ThrowsOnBadFileName) {
  std::string test_bin_file = "bad_file.txt";

  ASSERT_THAT(
    [&](){file_io::load_binary_block<double>(test_bin_file, 1, 1); },
    testing::ThrowsMessage<std::runtime_error>("load_binary_block - failed to open file: bad_file.txt")
  ); 
}

TEST(LoadBinaryBlock, ThrowsOnNonPositiveN_Rows) {
  std::vector<double> buff = {0.0,1.0,2.0,3.1,4.1,5.1,6.2,7.2,8.2};
  std::string test_bin_file = "test_3x3.bin";
  file_io::write_binary_block_to_file(test_bin_file.c_str(), buff.data(), buff.size());

  ASSERT_THAT(
    [&](){file_io::load_binary_block<double>(test_bin_file, 0, 3); },
    testing::ThrowsMessage<std::invalid_argument>("load_binary_block - n_rows must be positive")
  ); 

  std::remove(test_bin_file.c_str());
}

TEST(LoadBinaryBlock, ThrowsOnNonPositiveN_Cols) {
  std::vector<double> buff = {0.0,1.0,2.0,3.1,4.1,5.1,6.2,7.2,8.2};
  std::string test_bin_file = "test_3x3.bin";
  file_io::write_binary_block_to_file(test_bin_file.c_str(), buff.data(), buff.size());

  ASSERT_THAT(
    [&](){file_io::load_binary_block<double>(test_bin_file, 3, 0); },
    testing::ThrowsMessage<std::invalid_argument>("load_binary_block - n_cols must be positive")
  );

  std::remove(test_bin_file.c_str());
}

TEST(WriteBinaryBlockToFile, ThrowsOnBadFileName) {
  std::vector<double> buff = {0.0,1.0,2.0,3.1,4.1,5.1,6.2,7.2,8.2};
  std::string test_bin_file = "/bade_path/still_bad_path/test_3x3.bin";

    ASSERT_THAT(
    [&](){file_io::write_binary_block_to_file(test_bin_file.c_str(), buff.data(), buff.size()); },
    testing::ThrowsMessage<std::runtime_error>("write_binary_block_to_file - failed to open file: " + test_bin_file)
  );
}


TEST(Binary_IO, WriteBinaryAndLoadBinaryWork) {
  std::vector<double> buff = {0.0,1.0,2.0,3.1,4.1,5.1,6.2,7.2,8.2};
  std::string test_bin_file = "test_3x3.bin";

  file_io::write_binary_block_to_file(test_bin_file.c_str(), buff.data(), buff.size());
  auto data = file_io::load_binary_block<double>(test_bin_file, 3, 3);

  ASSERT_EQ(data.size(), buff.size());
  for (std::size_t i = 0; i < buff.size(); ++i) {
    EXPECT_DOUBLE_EQ(data[i], buff[i]);
  }
}

// TEST(Binary_IO, WritesBufferToFileAsync) {
//   std::vector<double> buff = {0.0,1.0,2.0,3.1,4.1,5.1,6.2,7.2,8.2};
//   std::string test_bin_file = "test_3x3_async.bin";
//   file_io::write_binary_block_to_file_async(test_bin_file.c_str(), buff.data(), buff.size());
// }

class FindBlockFilesTest : public ::testing::Test {
protected:
  fs::path temp_dir;

  void SetUp() override {
    temp_dir = fs::temp_directory_path() / fs::path("block_file_test");
    fs::create_directory(temp_dir);
  }

  void TearDown() override {
    fs::remove_all(temp_dir);  // Cleanup after test
  }

  void create_dummy_file(const std::string& filename) {
      std::ofstream f(temp_dir / filename, std::ios::binary);
      double dummy_data[4] = {1.0, 2.0, 3.0, 4.0};
      f.write(reinterpret_cast<char*>(dummy_data), sizeof(dummy_data));
  }

  void SetUpValidOnly() {
    create_dummy_file("test_block_0_0.bin");
    create_dummy_file("test_block_1_2.bin");
    create_dummy_file("test_block_6_3.bin");
  }

  void SetUpInvalidOnly() {
    create_dummy_file("test_block_bad.bin");
    create_dummy_file("another_block_bad.bin");
  }

  void SetUpMixed() {
    SetUpValidOnly();
    SetUpInvalidOnly();
  }
};

TEST_F(FindBlockFilesTest, ParsesValidBlockIndices) {
  SetUpValidOnly();

  auto blocks = file_io::find_block_files(temp_dir.string(), "test_block");

  std::set<std::pair<int, int>> expected = {
    {0, 0}, {1, 2}, {6, 3}
  };
  std::set<std::pair<int, int>> actual(blocks.begin(), blocks.end());

  EXPECT_EQ(actual, expected);
}

TEST_F(FindBlockFilesTest, IgnoresMalformedFiles) {
  SetUpInvalidOnly();

  auto blocks = file_io::find_block_files(temp_dir.string(), "test_block");

  EXPECT_TRUE(blocks.empty());
}

TEST_F(FindBlockFilesTest, SkipsInvalidAndParsesValid) {
  SetUpMixed();

  auto blocks = file_io::find_block_files(temp_dir.string(), "test_block");

  std::set<std::pair<int, int>> expected = {
    {0, 0}, {1, 2}, {6, 3}
  };
  std::set<std::pair<int, int>> actual(blocks.begin(), blocks.end());

  EXPECT_EQ(actual, expected);
}
