#include <gtest/gtest.h>
#include <gmock/gmock-matchers.h>
#include <filesystem>
#include <fstream>
#include <vector>
#include <limits>
#include "file_io/matrix_io.hpp"

namespace fs = std::filesystem;

class MergeBlocksColumnMajorTest : public ::testing::Test {
protected:
  fs::path temp_dir;
  std::string prefix = "test_block";

  void SetUp() override {
    temp_dir = fs::temp_directory_path() / "merge_column_major_test";
    fs::create_directories(temp_dir);

    // Create two test blocks:
    // test_block_0_1.bin (2x3)
    // test_block_1_0.bin (3x2)
    create_block_file(0, 1, {1, 2, 3, 4, 5, 6});  // column-major: 2x3
    create_block_file(1, 0, {10, 20, 30, 40, 50, 60});  // column-major: 3x2
  }

  void TearDown() override {
    fs::remove_all(temp_dir);
  }

  void create_block_file(int i, int j, const std::vector<double>& values) {
    std::ostringstream fname;
    fname << prefix << "_" << i << "_" << j << ".bin";
    fs::path filepath = temp_dir / fname.str();
    std::ofstream fout(filepath, std::ios::binary);
    fout.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(double));
  }
};

TEST(PrintColumnMajorMatrixDouble, ThrowsOnMatrixSizeMismatch) {
  const std::string file_name = "test_matrix.tsv";
  const std::vector<double> matrix = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
  const std::size_t n_rows = 5, n_cols = 3;

  ASSERT_THAT(
    [&](){file_io::print_column_major_matrix(file_name, matrix, n_rows, n_cols); },
    testing::ThrowsMessage<std::invalid_argument>("file_io::print_column_major_matrix - matrix size does not match dimensions: " + std::to_string(matrix.size()) + " vs. " + std::to_string(n_rows * n_cols))
  );
}

TEST(PrintColumnMajorMatrixDouble, ThrowsOnRowLabelSizeMismatch) {
  const std::string file_name = "test_matrix.tsv";
  const std::vector<double> matrix = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
  const std::size_t n_rows = 4, n_cols = 2;
  const std::vector<std::string> row_labels = {"row1", "row2", "row3"};

  ASSERT_THAT(
    [&](){file_io::print_column_major_matrix(file_name, matrix, n_rows, n_cols, row_labels); },
    testing::ThrowsMessage<std::invalid_argument>("file_io::print_column_major_matrix - row_lables size does not match number of rows")
  );
}

TEST(PrintColumnMajorMatrixDouble, ThrowsOColumnLabelSizeMismatch) {
  const std::string file_name = "test_matrix.tsv";
  const std::vector<double> matrix = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
  const std::size_t n_rows = 4, n_cols = 2;
  const std::vector<std::string> col_labels = {"row1", "row2", "row3"};

  ASSERT_THAT(
    [&](){file_io::print_column_major_matrix(file_name, matrix, n_rows, n_cols, {}, col_labels); },
    testing::ThrowsMessage<std::invalid_argument>("file_io::print_column_major_matrix - col_lables size does not match number of columns")
  );
}

TEST(PrintColumnMajorMatrixDouble, ThrowsOnBadFileName) {
  const std::string file_name = "/bad_dir/test_matrix.tsv";
  const std::vector<double> matrix = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
  const std::size_t n_rows = 4, n_cols = 2;

  ASSERT_THAT(
    [&](){file_io::print_column_major_matrix(file_name, matrix, n_rows, n_cols); },
    testing::ThrowsMessage<std::runtime_error>("file_io::print_column_major_matrix - could not open file for writing: " + file_name)
  );
}

TEST(PrintColumnMajorMatrixDouble, PrintsFixedNoLabels) {
  const std::string file_name = "test_matrix.tsv";
  const std::vector<double> matrix = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
  const std::size_t n_rows = 4, n_cols = 2;
  file_io::print_column_major_matrix(file_name, matrix, n_rows, n_cols);

  std::ifstream file(file_name);
  ASSERT_TRUE(file.is_open()) << "Could not open output file";

  std::string line;
  std::vector<std::string> lines;
  while (std::getline(file, line)) {
    lines.push_back(line);
  }

  file.close();
  std::remove(file_name.c_str());

  ASSERT_EQ(lines.size(), 4);
  EXPECT_EQ(lines[0], "1.000\t5.000");
  EXPECT_EQ(lines[1], "2.000\t6.000");
  EXPECT_EQ(lines[2], "3.000\t7.000");
  EXPECT_EQ(lines[3], "4.000\t8.000");
}

TEST(PrintColumnMajorMatrixDouble, PrintsScientificNoLabels) {
  const std::string file_name = "test_matrix.tsv";
  const std::vector<double> matrix = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
  const std::size_t n_rows = 4, n_cols = 2;
  file_io::print_column_major_matrix(file_name, matrix, n_rows, n_cols, {}, {}, 3, true);

  std::ifstream file(file_name);
  ASSERT_TRUE(file.is_open()) << "Could not open output file";

  std::string line;
  std::vector<std::string> lines;
  while (std::getline(file, line)) {
    lines.push_back(line);
  }

  file.close();
  std::remove(file_name.c_str());

  ASSERT_EQ(lines.size(), 4);
  EXPECT_EQ(lines[0], "1.000e+00\t5.000e+00");
  EXPECT_EQ(lines[1], "2.000e+00\t6.000e+00");
  EXPECT_EQ(lines[2], "3.000e+00\t7.000e+00");
  EXPECT_EQ(lines[3], "4.000e+00\t8.000e+00");
}

TEST(PrintColumnMajorMatrixDouble, PrintsScientificHighPrecision) {
  const std::string file_name = "test_matrix.tsv";
  const std::vector<double> matrix = {1.123456, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.7654321};
  const std::size_t n_rows = 4, n_cols = 2;
  file_io::print_column_major_matrix(file_name, matrix, n_rows, n_cols, {}, {}, 10, true);

  std::ifstream file(file_name);
  ASSERT_TRUE(file.is_open()) << "Could not open output file";

  std::string line;
  std::vector<std::string> lines;
  while (std::getline(file, line)) {
    lines.push_back(line);
  }

  file.close();
  std::remove(file_name.c_str());

  ASSERT_EQ(lines.size(), 4);
  EXPECT_EQ(lines[0], "1.1234560000e+00\t5.0000000000e+00");
  EXPECT_EQ(lines[1], "2.0000000000e+00\t6.0000000000e+00");
  EXPECT_EQ(lines[2], "3.0000000000e+00\t7.0000000000e+00");
  EXPECT_EQ(lines[3], "4.0000000000e+00\t8.7654321000e+00");
}

TEST(PrintColumnMajorMatrixDouble, PrintsFixedRowLabels) {
  const std::string file_name = "test_matrix.tsv";
  const std::vector<double> matrix = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
  const std::size_t n_rows = 4, n_cols = 2;
  const std::vector<std::string> row_labels = {"r1","r2","r3","r4"};
  file_io::print_column_major_matrix(file_name, matrix, n_rows, n_cols, row_labels);

  std::ifstream file(file_name);
  ASSERT_TRUE(file.is_open()) << "Could not open output file";

  std::string line;
  std::vector<std::string> lines;
  while (std::getline(file, line)) {
    lines.push_back(line);
  }

  file.close();
  std::remove(file_name.c_str());

  ASSERT_EQ(lines.size(), 4);
  EXPECT_EQ(lines[0], "r1\t1.000\t5.000");
  EXPECT_EQ(lines[1], "r2\t2.000\t6.000");
  EXPECT_EQ(lines[2], "r3\t3.000\t7.000");
  EXPECT_EQ(lines[3], "r4\t4.000\t8.000");
}

TEST(PrintColumnMajorMatrixDouble, PrintsFixedColLabels) {
  const std::string file_name = "test_matrix.tsv";
  const std::vector<double> matrix = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
  const std::size_t n_rows = 4, n_cols = 2;
  const std::vector<std::string> col_labels = {"c1","c2"};
  file_io::print_column_major_matrix(file_name, matrix, n_rows, n_cols, {}, col_labels);

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
  EXPECT_EQ(lines[0], "c1\tc2");
  EXPECT_EQ(lines[1], "1.000\t5.000");
  EXPECT_EQ(lines[2], "2.000\t6.000");
  EXPECT_EQ(lines[3], "3.000\t7.000");
  EXPECT_EQ(lines[4], "4.000\t8.000");
}

TEST(PrintColumnMajorMatrixDouble, PrintsFixedBothLabels) {
  const std::string file_name = "test_matrix.tsv";
  const std::vector<double> matrix = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
  const std::size_t n_rows = 4, n_cols = 2;
  const std::vector<std::string> row_labels = {"r1","r2","r3","r4"};
  const std::vector<std::string> col_labels = {"c1","c2"};
  file_io::print_column_major_matrix(file_name, matrix, n_rows, n_cols, row_labels, col_labels);

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
  EXPECT_EQ(lines[0], "INDEX\tc1\tc2");
  EXPECT_EQ(lines[1], "r1\t1.000\t5.000");
  EXPECT_EQ(lines[2], "r2\t2.000\t6.000");
  EXPECT_EQ(lines[3], "r3\t3.000\t7.000");
  EXPECT_EQ(lines[4], "r4\t4.000\t8.000");
}

TEST(PrintColumnMajorMatrixDouble, PrintsScientificNoLabelsTransposed) {
  const std::string file_name = "test_matrix.tsv";
  const std::vector<double> matrix = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
  const std::size_t n_rows = 4, n_cols = 2;
  file_io::print_column_major_matrix(file_name, matrix, n_rows, n_cols, {}, {}, 3, false, true);

  std::ifstream file(file_name);
  ASSERT_TRUE(file.is_open()) << "Could not open output file";

  std::string line;
  std::vector<std::string> lines;
  while (std::getline(file, line)) {
    lines.push_back(line);
  }

  file.close();
  std::remove(file_name.c_str());

  ASSERT_EQ(lines.size(), 2);
  EXPECT_EQ(lines[0], "1.000\t2.000\t3.000\t4.000");
  EXPECT_EQ(lines[1], "5.000\t6.000\t7.000\t8.000");
}

TEST(PrintColumnMajorMatrixDouble, PrintsFixedRowLabelsTransposed) {
  const std::string file_name = "test_matrix.tsv";
  const std::vector<double> matrix = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
  const std::size_t n_rows = 4, n_cols = 2;
  const std::vector<std::string> row_labels = {"r1","r2","r3","r4"};
  file_io::print_column_major_matrix(file_name, matrix, n_rows, n_cols, row_labels, {}, 3, false, true);

  std::ifstream file(file_name);
  ASSERT_TRUE(file.is_open()) << "Could not open output file";

  std::string line;
  std::vector<std::string> lines;
  while (std::getline(file, line)) {
    lines.push_back(line);
  }

  file.close();
  std::remove(file_name.c_str());

  ASSERT_EQ(lines.size(), 3);
  EXPECT_EQ(lines[0], "r1\tr2\tr3\tr4");
  EXPECT_EQ(lines[1], "1.000\t2.000\t3.000\t4.000");
  EXPECT_EQ(lines[2], "5.000\t6.000\t7.000\t8.000");
}

TEST(PrintColumnMajorMatrixDouble, PrintsFixedColLabelsTransposed) {
  const std::string file_name = "test_matrix.tsv";
  const std::vector<double> matrix = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
  const std::size_t n_rows = 4, n_cols = 2;
  const std::vector<std::string> col_labels = {"c1","c2"};
  file_io::print_column_major_matrix(file_name, matrix, n_rows, n_cols, {}, col_labels, 3, false, true);

  std::ifstream file(file_name);
  ASSERT_TRUE(file.is_open()) << "Could not open output file";

  std::string line;
  std::vector<std::string> lines;
  while (std::getline(file, line)) {
    lines.push_back(line);
  }

  file.close();
  std::remove(file_name.c_str());

  ASSERT_EQ(lines.size(), 2);
  EXPECT_EQ(lines[0], "c1\t1.000\t2.000\t3.000\t4.000");
  EXPECT_EQ(lines[1], "c2\t5.000\t6.000\t7.000\t8.000");
}

TEST(PrintColumnMajorMatrixDouble, PrintsFixedBothLabelsTransposed) {
  const std::string file_name = "test_matrix.tsv";
  const std::vector<double> matrix = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
  const std::size_t n_rows = 4, n_cols = 2;
  const std::vector<std::string> row_labels = {"r1","r2","r3","r4"};
  const std::vector<std::string> col_labels = {"c1","c2"};
  file_io::print_column_major_matrix(file_name, matrix, n_rows, n_cols, row_labels, col_labels, 3, false, true);

  std::ifstream file(file_name);
  ASSERT_TRUE(file.is_open()) << "Could not open output file";

  std::string line;
  std::vector<std::string> lines;
  while (std::getline(file, line)) {
    lines.push_back(line);
  }

  file.close();
  std::remove(file_name.c_str());

  ASSERT_EQ(lines.size(), 3);
  EXPECT_EQ(lines[0], "INDEX\tr1\tr2\tr3\tr4");
  EXPECT_EQ(lines[1], "c1\t1.000\t2.000\t3.000\t4.000");
  EXPECT_EQ(lines[2], "c2\t5.000\t6.000\t7.000\t8.000");
}

TEST(PrintColumnMajorMatrixBool, ThrowsOnMatrixSizeMismatch) {
  const std::string file_name = "test_matrix.tsv";
  const std::vector<bool> matrix = {true, true, true, false, true, false , true, false};
  const std::size_t n_rows = 5, n_cols = 3;

  ASSERT_THAT(
    [&](){file_io::print_column_major_matrix(file_name, matrix, n_rows, n_cols); },
    testing::ThrowsMessage<std::invalid_argument>("file_io::print_column_major_matrix - matrix size does not match dimensions: " + std::to_string(matrix.size()) + " vs. " + std::to_string(n_rows * n_cols))
  );
}

TEST(PrintColumnMajorMatrixBool, ThrowsOnRowLabelSizeMismatch) {
  const std::string file_name = "test_matrix.tsv";
  const std::vector<bool> matrix = {true, true, true, false, true, false , true, false};
  const std::size_t n_rows = 4, n_cols = 2;
  const std::vector<std::string> row_labels = {"row1", "row2", "row3"};

  ASSERT_THAT(
    [&](){file_io::print_column_major_matrix(file_name, matrix, n_rows, n_cols, row_labels); },
    testing::ThrowsMessage<std::invalid_argument>("file_io::print_column_major_matrix - row_lables size does not match number of rows")
  );
}

TEST(PrintColumnMajorMatrixBool, ThrowsOnColumnLabelSizeMismatch) {
  const std::string file_name = "test_matrix.tsv";
  const std::vector<bool> matrix = {true, true, true, false, true, false , true, false};
  const std::size_t n_rows = 4, n_cols = 2;
  const std::vector<std::string> col_labels = {"row1", "row2", "row3"};

  ASSERT_THAT(
    [&](){file_io::print_column_major_matrix(file_name, matrix, n_rows, n_cols, {}, col_labels); },
    testing::ThrowsMessage<std::invalid_argument>("file_io::print_column_major_matrix - col_lables size does not match number of columns")
  );
}

TEST(PrintColumnMajorMatrixBool, ThrowsOnBadFileName) {
  const std::string file_name = "/bad_dir/test_matrix.tsv";
  const std::vector<bool> matrix = {true, true, true, false, true, false , true, false};
  const std::size_t n_rows = 4, n_cols = 2;

  ASSERT_THAT(
    [&](){file_io::print_column_major_matrix(file_name, matrix, n_rows, n_cols); },
    testing::ThrowsMessage<std::runtime_error>("file_io::print_column_major_matrix - could not open file for writing: " + file_name)
  );
}

TEST(PrintColumnMajorMatrixBool, PrintNoLabels) {
  const std::string file_name = "test_matrix.tsv";
  const std::vector<bool> matrix = {true, true, false, false, true, false , true, false};
  const std::size_t n_rows = 4, n_cols = 2;
  file_io::print_column_major_matrix(file_name, matrix, n_rows, n_cols);

  std::ifstream file(file_name);
  ASSERT_TRUE(file.is_open()) << "Could not open output file";

  std::string line;
  std::vector<std::string> lines;
  while (std::getline(file, line)) {
    lines.push_back(line);
  }

  file.close();
  std::remove(file_name.c_str());

  ASSERT_EQ(lines.size(), 4);
  EXPECT_EQ(lines[0], "1\t1");
  EXPECT_EQ(lines[1], "1\t0");
  EXPECT_EQ(lines[2], "0\t1");
  EXPECT_EQ(lines[3], "0\t0");
}

TEST(PrintColumnMajorMatrixBool, PrintRowLabels) {
  const std::string file_name = "test_matrix.tsv";
  const std::vector<bool> matrix = {true, true, false, false, true, false , true, false};
  const std::size_t n_rows = 4, n_cols = 2;
  const std::vector<std::string> row_labels = {"r1","r2","r3","r4"};
  file_io::print_column_major_matrix(file_name, matrix, n_rows, n_cols, row_labels);

  std::ifstream file(file_name);
  ASSERT_TRUE(file.is_open()) << "Could not open output file";

  std::string line;
  std::vector<std::string> lines;
  while (std::getline(file, line)) {
    lines.push_back(line);
  }

  file.close();
  std::remove(file_name.c_str());

  ASSERT_EQ(lines.size(), 4);
  EXPECT_EQ(lines[0], "r1\t1\t1");
  EXPECT_EQ(lines[1], "r2\t1\t0");
  EXPECT_EQ(lines[2], "r3\t0\t1");
  EXPECT_EQ(lines[3], "r4\t0\t0");
}

TEST(PrintColumnMajorMatrixBool, PrintColLabels) {
  const std::string file_name = "test_matrix.tsv";
  const std::vector<bool> matrix = {true, true, false, false, true, false , true, false};
  const std::size_t n_rows = 4, n_cols = 2;
  const std::vector<std::string> col_labels = {"c1","c2"};
  file_io::print_column_major_matrix(file_name, matrix, n_rows, n_cols, {}, col_labels);

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
  EXPECT_EQ(lines[0], "c1\tc2");
  EXPECT_EQ(lines[1], "1\t1");
  EXPECT_EQ(lines[2], "1\t0");
  EXPECT_EQ(lines[3], "0\t1");
  EXPECT_EQ(lines[4], "0\t0");
}

TEST(PrintColumnMajorMatrixBool, PrintBothLabels) {
  const std::string file_name = "test_matrix.tsv";
  const std::vector<bool> matrix = {true, true, false, false, true, false , true, false};
  const std::size_t n_rows = 4, n_cols = 2;
  const std::vector<std::string> row_labels = {"r1","r2","r3","r4"};
  const std::vector<std::string> col_labels = {"c1","c2"};
  file_io::print_column_major_matrix(file_name, matrix, n_rows, n_cols, row_labels, col_labels);

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
  EXPECT_EQ(lines[0], "INDEX\tc1\tc2");
  EXPECT_EQ(lines[1], "r1\t1\t1");
  EXPECT_EQ(lines[2], "r2\t1\t0");
  EXPECT_EQ(lines[3], "r3\t0\t1");
  EXPECT_EQ(lines[4], "r4\t0\t0");
}

TEST(PrintColumnMajorMatrixBool, PrintNoLabelsTransposed) {
  const std::string file_name = "test_matrix.tsv";
  const std::vector<bool> matrix = {true, true, false, false, true, false , true, false};
  const std::size_t n_rows = 4, n_cols = 2;
  file_io::print_column_major_matrix(file_name, matrix, n_rows, n_cols, {}, {}, true);

  std::ifstream file(file_name);
  ASSERT_TRUE(file.is_open()) << "Could not open output file";

  std::string line;
  std::vector<std::string> lines;
  while (std::getline(file, line)) {
    lines.push_back(line);
  }

  file.close();
  std::remove(file_name.c_str());

  ASSERT_EQ(lines.size(), 2);
  EXPECT_EQ(lines[0], "1\t1\t0\t0");
  EXPECT_EQ(lines[1], "1\t0\t1\t0");
}

TEST(PrintColumnMajorMatrixBool, PrintRowLabelsTransposed) {
    const std::string file_name = "test_matrix.tsv";
  const std::vector<bool> matrix = {true, true, false, false, true, false , true, false};
  const std::size_t n_rows = 4, n_cols = 2;
  const std::vector<std::string> row_labels = {"r1","r2","r3","r4"};
  file_io::print_column_major_matrix(file_name, matrix, n_rows, n_cols, row_labels, {}, true);

  std::ifstream file(file_name);
  ASSERT_TRUE(file.is_open()) << "Could not open output file";

  std::string line;
  std::vector<std::string> lines;
  while (std::getline(file, line)) {
    lines.push_back(line);
  }

  file.close();
  std::remove(file_name.c_str());

  ASSERT_EQ(lines.size(), 3);
  EXPECT_EQ(lines[0], "r1\tr2\tr3\tr4");
  EXPECT_EQ(lines[1], "1\t1\t0\t0");
  EXPECT_EQ(lines[2], "1\t0\t1\t0");
}

TEST(PrintColumnMajorMatrixBool, PrintColLabelsTransposed) {
  const std::string file_name = "test_matrix.tsv";
  const std::vector<bool> matrix = {true, true, false, false, true, false , true, false};
  const std::size_t n_rows = 4, n_cols = 2;
  const std::vector<std::string> col_labels = {"c1","c2"};
  file_io::print_column_major_matrix(file_name, matrix, n_rows, n_cols, {}, col_labels, true);

  std::ifstream file(file_name);
  ASSERT_TRUE(file.is_open()) << "Could not open output file";

  std::string line;
  std::vector<std::string> lines;
  while (std::getline(file, line)) {
    lines.push_back(line);
  }

  file.close();
  std::remove(file_name.c_str());

  ASSERT_EQ(lines.size(), 2);
  EXPECT_EQ(lines[0], "c1\t1\t1\t0\t0");
  EXPECT_EQ(lines[1], "c2\t1\t0\t1\t0");
}

TEST(PrintColumnMajorMatrixBool, PrintBothLabelsTransposed) {
  const std::string file_name = "test_matrix.tsv";
  const std::vector<bool> matrix = {true, true, false, false, true, false , true, false};
  const std::size_t n_rows = 4, n_cols = 2;
  const std::vector<std::string> row_labels = {"r1","r2","r3","r4"};
  const std::vector<std::string> col_labels = {"c1","c2"};
  file_io::print_column_major_matrix(file_name, matrix, n_rows, n_cols, row_labels, col_labels, true);

  std::ifstream file(file_name);
  ASSERT_TRUE(file.is_open()) << "Could not open output file";

  std::string line;
  std::vector<std::string> lines;
  while (std::getline(file, line)) {
    lines.push_back(line);
  }

  file.close();
  std::remove(file_name.c_str());

  ASSERT_EQ(lines.size(), 3);
  EXPECT_EQ(lines[0], "INDEX\tr1\tr2\tr3\tr4");
  EXPECT_EQ(lines[1], "c1\t1\t1\t0\t0");
  EXPECT_EQ(lines[2], "c2\t1\t0\t1\t0");
}

TEST(PrintColumnMajorDistanceMatrix, ThrowsOnMatrixSizeMismatch) {
  const std::string file_name = "test_matrix.tsv";
  const std::vector<double> matrix = {1.0, 2.0, 3.0, 2.0, 4.0, 5.0, 3.0, 5.0, 6.0};
  const std::size_t dim = 4;

  ASSERT_THAT(
    [&](){file_io::print_column_major_distance_matrix(file_name, matrix, dim); },
    testing::ThrowsMessage<std::invalid_argument>("file_io::print_column_major_distance_matrix - matrix size does not match dimensions")
  );
}

TEST(PrintColumnMajorDistanceMatrix, ThrowsOnLabelSizeMismatch) {
  const std::string file_name = "test_matrix.tsv";
  const std::vector<double> matrix = {1.0, 2.0, 3.0, 2.0, 4.0, 5.0, 3.0, 5.0, 6.0};
  const std::size_t dim = 3;
  const std::vector<std::string> labels = {"node1", "node2"};

  ASSERT_THAT(
    [&](){file_io::print_column_major_distance_matrix(file_name, matrix, dim, labels); },
    testing::ThrowsMessage<std::invalid_argument>("file_io::print_column_major_distance_matrix - lables size does not match dimension")
  );
}

TEST(PrintColumnMajorDistanceMatrix, ThrowsOnBadFileName) {
  const std::string file_name = "/bad_dir/test_matrix.tsv";
  const std::vector<double> matrix = {1.0, 2.0, 3.0, 2.0, 4.0, 5.0, 3.0, 5.0, 6.0};
  const std::size_t dim = 3;

  ASSERT_THAT(
    [&](){file_io::print_column_major_distance_matrix(file_name, matrix, dim); },
    testing::ThrowsMessage<std::runtime_error>("file_io::print_column_major_distance_matrix - could not open file for writing: " + file_name)
  );
}

TEST(PrintColumnMajorDistanceMatrix, PrintsNoLabelsFixed) {
  const std::string file_name = "test_matrix.tsv";
  const std::vector<double> matrix = {1.0, 2.0, 3.0, 2.0, 4.0, 5.0, 3.0, 5.0, 6.0};
  const std::size_t dim = 3;

  file_io::print_column_major_distance_matrix(file_name, matrix, dim);

  std::ifstream file(file_name);
  ASSERT_TRUE(file.is_open()) << "Could not open output file";

  std::string line;
  std::vector<std::string> lines;
  while (std::getline(file, line)) {
    lines.push_back(line);
  }

  file.close();
  std::remove(file_name.c_str());

  ASSERT_EQ(lines.size(), 3);
  EXPECT_EQ(lines[0], "1.000\t2.000\t3.000");
  EXPECT_EQ(lines[1], "2.000\t4.000\t5.000");
  EXPECT_EQ(lines[2], "3.000\t5.000\t6.000");
}

TEST(PrintColumnMajorDistanceMatrix, PrintsNoLabelsHighPrecision) {
  const std::string file_name = "test_matrix.tsv";
  const std::vector<double> matrix = {1.123456, 2.0, 3.0, 2.0, 4.0, 5.0, 3.0, 5.0, 6.0};
  const std::size_t dim = 3;

  file_io::print_column_major_distance_matrix(file_name, matrix, dim, {}, 5);

  std::ifstream file(file_name);
  ASSERT_TRUE(file.is_open()) << "Could not open output file";

  std::string line;
  std::vector<std::string> lines;
  while (std::getline(file, line)) {
    lines.push_back(line);
  }

  file.close();
  std::remove(file_name.c_str());

  ASSERT_EQ(lines.size(), 3);
  EXPECT_EQ(lines[0], "1.12346\t2.00000\t3.00000");
  EXPECT_EQ(lines[1], "2.00000\t4.00000\t5.00000");
  EXPECT_EQ(lines[2], "3.00000\t5.00000\t6.00000");
}

TEST(PrintColumnMajorDistanceMatrix, PrintsNoLabelsScientific) {
  const std::string file_name = "test_matrix.tsv";
  const std::vector<double> matrix = {1.0, 2.0, 3.0, 2.0, 4.0, 5.0, 3.0, 5.0, 6.0};
  const std::size_t dim = 3;

  file_io::print_column_major_distance_matrix(file_name, matrix, dim, {}, 3, true);

  std::ifstream file(file_name);
  ASSERT_TRUE(file.is_open()) << "Could not open output file";

  std::string line;
  std::vector<std::string> lines;
  while (std::getline(file, line)) {
    lines.push_back(line);
  }

  file.close();
  std::remove(file_name.c_str());

  ASSERT_EQ(lines.size(), 3);
  EXPECT_EQ(lines[0], "1.000e+00\t2.000e+00\t3.000e+00");
  EXPECT_EQ(lines[1], "2.000e+00\t4.000e+00\t5.000e+00");
  EXPECT_EQ(lines[2], "3.000e+00\t5.000e+00\t6.000e+00");
}

TEST(PrintColumnMajorDistanceMatrix, PrintsLabelsFixed) {
  const std::string file_name = "test_matrix.tsv";
  const std::vector<double> matrix = {1.0, 2.0, 3.0, 2.0, 4.0, 5.0, 3.0, 5.0, 6.0};
  const std::size_t dim = 3;
  const std::vector<std::string> labels = {"n1","n2","n3"};

  file_io::print_column_major_distance_matrix(file_name, matrix, dim, labels);

  std::ifstream file(file_name);
  ASSERT_TRUE(file.is_open()) << "Could not open output file";

  std::string line;
  std::vector<std::string> lines;
  while (std::getline(file, line)) {
    lines.push_back(line);
  }

  file.close();
  std::remove(file_name.c_str());

  ASSERT_EQ(lines.size(), 4);
  EXPECT_EQ(lines[0], "INDEX\tn1\tn2\tn3");
  EXPECT_EQ(lines[1], "n1\t1.000\t2.000\t3.000");
  EXPECT_EQ(lines[2], "n2\t2.000\t4.000\t5.000");
  EXPECT_EQ(lines[3], "n3\t3.000\t5.000\t6.000");
}

TEST(PrintColumnMajorDistanceMatrix, PrintsLabelsUppper) {
  const std::string file_name = "test_matrix.tsv";
  const std::vector<double> matrix = {1.0, 2.0, 3.0, 2.0, 4.0, 5.0, 3.0, 5.0, 6.0};
  const std::size_t dim = 3;
  const std::vector<std::string> labels = {"n1","n2","n3"};

  file_io::print_column_major_distance_matrix(file_name, matrix, dim, labels, 3, false, file_io::DIST_MATRIX_MODE::UPPER);

  std::ifstream file(file_name);
  ASSERT_TRUE(file.is_open()) << "Could not open output file";

  std::string line;
  std::vector<std::string> lines;
  while (std::getline(file, line)) {
    lines.push_back(line);
  }

  file.close();
  std::remove(file_name.c_str());

  ASSERT_EQ(lines.size(), 4);
  EXPECT_EQ(lines[0], "INDEX\tn1\tn2\tn3");
  EXPECT_EQ(lines[1], "n1\tNA\t2.000\t3.000");
  EXPECT_EQ(lines[2], "n2\tNA\tNA\t5.000");
  EXPECT_EQ(lines[3], "n3\tNA\tNA\tNA");
}

TEST(PrintColumnMajorDistanceMatrix, PrintsLabelsLower) {
    const std::string file_name = "test_matrix.tsv";
  const std::vector<double> matrix = {1.0, 2.0, 3.0, 2.0, 4.0, 5.0, 3.0, 5.0, 6.0};
  const std::size_t dim = 3;
  const std::vector<std::string> labels = {"n1","n2","n3"};

  file_io::print_column_major_distance_matrix(file_name, matrix, dim, labels, 3, false, file_io::DIST_MATRIX_MODE::LOWER);

  std::ifstream file(file_name);
  ASSERT_TRUE(file.is_open()) << "Could not open output file";

  std::string line;
  std::vector<std::string> lines;
  while (std::getline(file, line)) {
    lines.push_back(line);
  }

  file.close();
  std::remove(file_name.c_str());

  ASSERT_EQ(lines.size(), 4);
  EXPECT_EQ(lines[0], "INDEX\tn1\tn2\tn3");
  EXPECT_EQ(lines[1], "n1\tNA\tNA\tNA");
  EXPECT_EQ(lines[2], "n2\t2.000\tNA\tNA");
  EXPECT_EQ(lines[3], "n3\t3.000\t5.000\tNA");
}

TEST_F(MergeBlocksColumnMajorTest, MergesCorrectlyIntoFlatMatrix) {
  // Assume rank 0 has 2 vectors, rank 1 has 3
  std::vector<std::size_t> N_ranks = {2, 3};

  std::vector<double> matrix;
  file_io::read_matrix_from_binary_blocks(matrix, temp_dir.string(), prefix, N_ranks);

  // Full matrix should be 5x5 column-major
  ASSERT_EQ(matrix.size(), 25);

  const double nan = std::numeric_limits<double>::quiet_NaN();

  auto at = [&](int row, int col) -> double {
    return matrix[col * 5 + row];  // column-major access
  };

  // Check test_block_0_1.bin (2x3), inserted at [0:2, 2:5]
  EXPECT_DOUBLE_EQ(at(0, 2), 1.0);
  EXPECT_DOUBLE_EQ(at(1, 2), 2.0);
  EXPECT_DOUBLE_EQ(at(0, 3), 3.0);
  EXPECT_DOUBLE_EQ(at(1, 3), 4.0);
  EXPECT_DOUBLE_EQ(at(0, 4), 5.0);
  EXPECT_DOUBLE_EQ(at(1, 4), 6.0);

  // Check test_block_1_0.bin (3x2), inserted at [2:5, 0:2]
  EXPECT_DOUBLE_EQ(at(2, 0), 10.0);
  EXPECT_DOUBLE_EQ(at(3, 0), 20.0);
  EXPECT_DOUBLE_EQ(at(4, 0), 30.0);
  EXPECT_DOUBLE_EQ(at(2, 1), 40.0);
  EXPECT_DOUBLE_EQ(at(3, 1), 50.0);
  EXPECT_DOUBLE_EQ(at(4, 1), 60.0);

  // Check a few unused positions are still NaN
  EXPECT_TRUE(std::isnan(at(0, 0)));
  EXPECT_TRUE(std::isnan(at(4, 4)));
}
