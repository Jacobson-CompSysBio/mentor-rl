#include <gtest/gtest.h>
#include <gmock/gmock-matchers.h>
#include "file_io/vector_io.hpp"
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>

// Helper to create temporary test file
void write_test_file(const std::string& filename, const std::string& content) {
  std::ofstream out(filename);
  out << content;
  out.close();
}

TEST(PrintVector, ThrowsOnBadFileName) {
  std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
  const std::string file_name = "/bad_dir/bad_filename.tsv";
  ASSERT_THAT(
    [&](){file_io::print_vector(data, file_name); },
    testing::ThrowsMessage<std::runtime_error>("file_io::print_vector - unable to open file " + file_name)
  );
}

TEST(PrintVector, PrintsRowVecFloatingPointAsScientific) {
  std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
  const std::string file_name = "test_vector.tsv";

  file_io::print_vector(data, file_name, false, 3, true);

  std::ifstream file(file_name);
  ASSERT_TRUE(file.is_open()) << "Could not open output file";

  std::string line;
  std::vector<std::string> lines;
  while (std::getline(file, line)) {
    lines.push_back(line);
  }

  file.close();
  std::remove(file_name.c_str());

  ASSERT_EQ(lines.size(), 1);
  EXPECT_EQ(lines[0], "1.000e+00\t2.000e+00\t3.000e+00\t4.000e+00\t5.000e+00");
}

TEST(PrintVector, PrintsRowVecFloatingPointAsFixed) {
  std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 0.005};
  const std::string file_name = "test_vector.tsv";

  file_io::print_vector(data, file_name, false, 4, false);

  std::ifstream file(file_name);
  ASSERT_TRUE(file.is_open()) << "Could not open output file";

  std::string line;
  std::vector<std::string> lines;
  while (std::getline(file, line)) {
    lines.push_back(line);
  }

  file.close();
  std::remove(file_name.c_str());

  ASSERT_EQ(lines.size(), 1);
  EXPECT_EQ(lines[0], "1.0000\t2.0000\t3.0000\t4.0000\t0.0050");
}

TEST(PrintVector, PrintsRowVecAsCsv) {
  std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 0.005};
  const std::string file_name = "test_vector.tsv";

  file_io::print_vector(data, file_name, false, 4, false, ',');

  std::ifstream file(file_name);
  ASSERT_TRUE(file.is_open()) << "Could not open output file";

  std::string line;
  std::vector<std::string> lines;
  while (std::getline(file, line)) {
    lines.push_back(line);
  }

  file.close();
  std::remove(file_name.c_str());

  ASSERT_EQ(lines.size(), 1);
  EXPECT_EQ(lines[0], "1.0000,2.0000,3.0000,4.0000,0.0050");
}

TEST(PrintVector, PrintsColumnVec) {
  std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 0.005};
  const std::string file_name = "test_vector.tsv";

  file_io::print_vector(data, file_name, true, 4, false, ',');

  std::ifstream file(file_name);
  ASSERT_TRUE(file.is_open()) << "Could not open output file";

  std::string line;
  std::vector<std::string> lines;
  while (std::getline(file, line)) {
    lines.push_back(line);
  }

  file.close();
  std::remove(file_name.c_str());

  ASSERT_EQ(lines.size(), 5);
  EXPECT_EQ(lines[0], "1.0000");
  EXPECT_EQ(lines[1], "2.0000");
  EXPECT_EQ(lines[2], "3.0000");
  EXPECT_EQ(lines[3], "4.0000");
  EXPECT_EQ(lines[4], "0.0050");
}

TEST(ReadVector, ThrowsOnBadFileName) {
  const std::string file_name = "/bad_dir/bad_filename.tsv";
  ASSERT_THAT(
    [&](){auto data = file_io::read_vector<double>(file_name); },
    testing::ThrowsMessage<std::runtime_error>("file_io::read_vector - unable to open file " + file_name)
  );
}

TEST(ReadVector, ReadsColumnVec) {
  std::string file_name = "test_vector.tsv";
  write_test_file(file_name,
    "1.1\n"
    "1.2\n"
    "1.3\n"
    "2.4\n"
    "3.5\n"
  );

  auto output = file_io::read_vector<double>(file_name, true);
  std::remove(file_name.c_str());

  ASSERT_EQ(output.size(), 5);
  EXPECT_EQ(output[0], 1.1);
  EXPECT_EQ(output[1], 1.2);
  EXPECT_EQ(output[2], 1.3);
  EXPECT_EQ(output[3], 2.4);
  EXPECT_EQ(output[4], 3.5);
}

TEST(ReadVector, ReadsColumnVecFromCsv) {
  std::string file_name = "test_vector.tsv";
  write_test_file(file_name,
    "1.1\n"
    "1.2\n"
    "1.3\n"
    "2.4\n"
    "3.5\n"
  );

  auto output = file_io::read_vector<double>(file_name, true, ',');
  std::remove(file_name.c_str());

  ASSERT_EQ(output.size(), 5);
  EXPECT_EQ(output[0], 1.1);
  EXPECT_EQ(output[1], 1.2);
  EXPECT_EQ(output[2], 1.3);
  EXPECT_EQ(output[3], 2.4);
  EXPECT_EQ(output[4], 3.5);
}

TEST(ReadVector, ReadsTpwVecFromCsv) {
  std::string file_name = "test_vector.tsv";
  write_test_file(file_name,
    "1.1,1.2,1.3,2.4,3.5\n"
  );

  auto output = file_io::read_vector<double>(file_name, false, ',');
  std::remove(file_name.c_str());

  ASSERT_EQ(output.size(), 5);
  EXPECT_EQ(output[0], 1.1);
  EXPECT_EQ(output[1], 1.2);
  EXPECT_EQ(output[2], 1.3);
  EXPECT_EQ(output[3], 2.4);
  EXPECT_EQ(output[4], 3.5);
}

TEST(ReadVector, ReadsColumnVecToInt) {
  std::string file_name = "test_vector.tsv";
  write_test_file(file_name,
    "1\n2\n3\n4\n5\n"
  );

  auto output = file_io::read_vector<int>(file_name, true);
  std::remove(file_name.c_str());

  ASSERT_EQ(output.size(), 5);
  EXPECT_EQ(output[0], 1);
  EXPECT_EQ(output[1], 2);
  EXPECT_EQ(output[2], 3);
  EXPECT_EQ(output[3], 4);
  EXPECT_EQ(output[4], 5);
}