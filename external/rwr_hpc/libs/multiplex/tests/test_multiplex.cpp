#include <gtest/gtest.h>
#include <gmock/gmock-matchers.h>
#include <gmock/gmock.h>

#include <fstream>
#include <iostream>
#include <multiplex/Multiplex.hpp>

// Helper to create temporary test file
void write_test_file(const std::string& filename, const std::string& content) {
  std::ofstream out(filename);
  out << content;
  out.close();
}

class MultiplexProtected : public Multiplex {
  using Multiplex::Multiplex;

  public:
    void read_flist(const std::string& file_name, const bool has_headers) {
      Multiplex::read_flist(file_name, has_headers);
    }
    std::vector<bool> create_local_layer_list(const std::vector<bool> &layer_list) const {
      return Multiplex::create_local_layer_list(layer_list);
    }
    std::vector<std::string> create_local_label_list(const std::vector<std::string> &label_list) const {
      return Multiplex::create_local_label_list(label_list);
    }
  
};

TEST(TestMultiplex, DefualtConstructor) {
  Multiplex mp;
  EXPECT_EQ(mp.n_nodes(), 0);
  EXPECT_EQ(mp.n_layers(), 0);
  const auto actual_nodes = mp.get_nodes();
  EXPECT_EQ(actual_nodes.size(), 0);
}

TEST(TestMultiplex, ReadFlistThrowsIfFileCannotBeOpened) {
  MultiplexProtected mp;
  const std::string flist = "bad_flist.txt";

  ASSERT_THAT(
    [&](){mp.read_flist(flist, false); },
    testing::ThrowsMessage<std::invalid_argument>("Multiplex::read_flist - flist could not be opened")
  );
}

TEST(TestMultiplex, ReadFlistThrowsIfFlistIsEmpty) {
  MultiplexProtected mp;
  const std::string flist = "empty_flist.txt";
  write_test_file(flist,"");

  ASSERT_THAT(
    [&](){mp.read_flist(flist, false); },
    testing::ThrowsMessage<std::runtime_error>("Multiplex::read_flist - flist is empty")
  );

  std::remove(flist.c_str());
}

TEST(TestMultiplex, ReadFlistThrowsIfFlistHasLessThanTwoColumns) {
  const std::string edgelist1_filename = "edgelist1.txt";
  write_test_file(edgelist1_filename,
    "A\tB\n"
    "A\tC\n"
    "B\tC\n"
    "C\tD\n"
  );
  
  const std::string edgelist2_filename = "edgelist2.txt";
  write_test_file(edgelist2_filename,
    "A\tB\n"
    "B\tC\n"
    "B\tD\n"
    "C\tD\n"
  );

  const std::string filename = "test_flist.txt";
  write_test_file(filename,
    "edgelist1.txt\n"
    "edgelist2.txt\n"
  );

  MultiplexProtected mp;
  ASSERT_THAT(
    [&](){mp.read_flist(filename, false); },
    testing::ThrowsMessage<std::runtime_error>("Multiplex::read_flist - Line 0 has fewer than 2 tab-separated columns")
  );

  std::remove(edgelist1_filename.c_str());
  std::remove(edgelist2_filename.c_str());
  std::remove(filename.c_str());
}

TEST(TestMultiplex, ReadFlistParsesNoHeaders) {
  const std::string edgelist1_filename = "edgelist1.txt";
  write_test_file(edgelist1_filename,
    "A\tB\n"
    "A\tC\n"
    "B\tC\n"
    "C\tD\n"
  );

  const std::string edgelist2_filename = "edgelist2.txt";
  write_test_file(edgelist2_filename,
    "A\tB\n"
    "B\tC\n"
    "B\tD\n"
    "C\tE\n"
  );

  const std::string filename = "test_flist.txt";
  write_test_file(filename,
    "edgelist1.txt\tlayer1\n"
    "edgelist2.txt\tlayer2\n"
  );

  MultiplexProtected mp;
  mp.read_flist(filename, false);
  auto actual_nodes = mp.get_nodes();
  auto actual_layer_names = mp.get_layer_names();

  std::vector<std::string> expected_nodes = {"A","B","C","D","E"};
  std::vector<std::string> expected_layer_names = {"layer1","layer2"};

  EXPECT_EQ(mp.n_layers(), 2);
  EXPECT_EQ(mp.n_nodes(), 5);
  EXPECT_EQ(actual_nodes, expected_nodes);
  EXPECT_EQ(actual_layer_names, expected_layer_names);

  std::remove(edgelist1_filename.c_str());
  std::remove(edgelist2_filename.c_str());
  std::remove(filename.c_str());
}

TEST(TestMultiplex, ReadFlistParsesWithHeaders) {
  const std::string edgelist1_filename = "edgelist1.txt";
  write_test_file(edgelist1_filename,
    "src\ttgt\n"
    "A\tB\n"
    "A\tC\n"
    "B\tC\n"
    "C\tD\n"
  );

  const std::string edgelist2_filename = "edgelist2.txt";
  write_test_file(edgelist2_filename,
    "src\ttgt\n"
    "A\tB\n"
    "B\tC\n"
    "B\tD\n"
    "C\tE\n"
  );

  const std::string filename = "test_flist.txt";
  write_test_file(filename,
    "edgelist1.txt\tlayer1\n"
    "edgelist2.txt\tlayer2\n"
  );

  MultiplexProtected mp;
  mp.read_flist(filename, true);
  auto actual_nodes = mp.get_nodes();
  auto actual_layer_names = mp.get_layer_names();

  std::vector<std::string> expected_nodes = {"A","B","C","D","E"};
  std::vector<std::string> expected_layer_names = {"layer1","layer2"};

  EXPECT_EQ(mp.n_layers(), 2);
  EXPECT_EQ(mp.n_nodes(), 5);
  EXPECT_EQ(actual_nodes, expected_nodes);
  EXPECT_EQ(actual_layer_names, expected_layer_names);

  std::remove(edgelist1_filename.c_str());
  std::remove(edgelist2_filename.c_str());
  std::remove(filename.c_str());
}

TEST(TestMultiplex, GetNodesReturnsExpectedNodes) {
  const std::string edgelist1_filename = "edgelist1.txt";
  write_test_file(edgelist1_filename,
    "F\tG\n"
    "A\tB\n"
    "A\tC\n"
    "B\tC\n"
    "C\tD\n"
  );

  const std::string edgelist2_filename = "edgelist2.txt";
  write_test_file(edgelist2_filename,
    "A\tB\n"
    "B\tC\n"
    "B\tD\n"
    "C\tE\n"
  );

  const std::string filename = "test_flist.txt";
  write_test_file(filename,
    "edgelist1.txt\tlayer1\n"
    "edgelist2.txt\tlayer2\n"
  );

  Multiplex mp(filename);

  auto actual_nodes = mp.get_nodes();
  std::vector<std::string> expected_nodes = {"A","B","C","D","E","F","G"};

  EXPECT_EQ(actual_nodes, expected_nodes);

  std::remove(edgelist1_filename.c_str());
  std::remove(edgelist2_filename.c_str());
  std::remove(filename.c_str());
}

TEST(TestMultiplex, GetLayerNamesReturnsExpectedNames) {
  const std::string edgelist1_filename = "edgelist1.txt";
  write_test_file(edgelist1_filename,
    "A\tB\n"
    "A\tC\n"
    "B\tC\n"
    "C\tD\n"
  );

  const std::string edgelist2_filename = "edgelist2.txt";
  write_test_file(edgelist2_filename,
    "A\tB\n"
    "B\tC\n"
    "B\tD\n"
    "C\tE\n"
  );

  const std::string edgelist3_filename = "edgelist3.txt";
  write_test_file(edgelist3_filename,
    "B\tC\n"
    "C\tE\n"
  );

  const std::string filename = "test_flist.txt";
  write_test_file(filename,
    "edgelist1.txt\tlayer1\n"
    "edgelist2.txt\tlayer2\n"
    "edgelist3.txt\tlayer3\n"
  );

  Multiplex mp(filename);

  std::remove(edgelist1_filename.c_str());
  std::remove(edgelist2_filename.c_str());
  std::remove(edgelist3_filename.c_str());
  std::remove(filename.c_str());

  auto actual_layer_names = mp.get_layer_names();
  std::vector<std::string> expected_layer_names = {"layer1", "layer2", "layer3"};

  EXPECT_EQ(actual_layer_names, expected_layer_names);
}

TEST(TestMultiplex, GetNodesByLayerReturnsExpectedValues) {
  const std::string edgelist1_filename = "edgelist1.txt";
  write_test_file(edgelist1_filename,
    "A\tB\n"
    "A\tC\n"
    "B\tC\n"
    "C\tD\n"
  );

  const std::string edgelist2_filename = "edgelist2.txt";
  write_test_file(edgelist2_filename,
    "A\tB\n"
    "B\tC\n"
    "B\tD\n"
    "C\tE\n"
  );

  const std::string edgelist3_filename = "edgelist3.txt";
  write_test_file(edgelist3_filename,
    "B\tC\n"
    "C\tE\n"
  );

  const std::string filename = "test_flist.txt";
  write_test_file(filename,
    "edgelist1.txt\tlayer1\n"
    "edgelist2.txt\tlayer2\n"
    "edgelist3.txt\tlayer3\n"
  );

  Multiplex mp(filename);

  std::remove(edgelist1_filename.c_str());
  std::remove(edgelist2_filename.c_str());
  std::remove(edgelist3_filename.c_str());
  std::remove(filename.c_str());

  auto actual_nodes_by_layer = mp.get_nodes_by_layer();
  std::vector<bool> expected_nodes_by_layer = {true, true, false,
                                               true, true, true,
                                               true, true, true,
                                               true, true, false,
                                               false, true, true};
  EXPECT_EQ(actual_nodes_by_layer, expected_nodes_by_layer);
}

TEST(TestMultiplex, CreateLocalLayerListThrowsOnSizeMismatch) {
  const std::string edgelist1_filename = "edgelist1.txt";
  write_test_file(edgelist1_filename,
    "A\tB\n"
    "A\tC\n"
    "B\tC\n"
    "C\tD\n"
  );

  const std::string edgelist2_filename = "edgelist2.txt";
  write_test_file(edgelist2_filename,
    "A\tB\n"
    "B\tC\n"
    "B\tD\n"
    "C\tE\n"
  );

  const std::string filename = "test_flist.txt";
  write_test_file(filename,
    "edgelist1.txt\tlayer1\n"
    "edgelist2.txt\tlayer2\n"
  );

  MultiplexProtected mp(filename, false);

  ASSERT_THAT(
    [&](){mp.create_local_layer_list({true}); },
    testing::ThrowsMessage<std::invalid_argument>("Multiplex::create_local_layer_list - layer_list length does not match number of layers")
  );

  std::remove(edgelist1_filename.c_str());
  std::remove(edgelist2_filename.c_str());
  std::remove(filename.c_str());
}

TEST(TestMultiplex, CreateLocalLayerListEmptyInput) {
  const std::string edgelist1_filename = "edgelist1.txt";
  write_test_file(edgelist1_filename,
    "A\tB\n"
    "A\tC\n"
    "B\tC\n"
    "C\tD\n"
  );

  const std::string edgelist2_filename = "edgelist2.txt";
  write_test_file(edgelist2_filename,
    "A\tB\n"
    "B\tC\n"
    "B\tD\n"
    "C\tE\n"
  );

  const std::string filename = "test_flist.txt";
  write_test_file(filename,
    "edgelist1.txt\tlayer1\n"
    "edgelist2.txt\tlayer2\n"
  );

  MultiplexProtected mp(filename, false);

  auto local_layer_list = mp.create_local_layer_list({});
  ASSERT_EQ(local_layer_list.size(), 2);
  EXPECT_EQ(local_layer_list[0], true);
  EXPECT_EQ(local_layer_list[1], true);

  std::remove(edgelist1_filename.c_str());
  std::remove(edgelist2_filename.c_str());
  std::remove(filename.c_str());
}

TEST(TestMultiplex, CreateLocalLayerListNonemptyInput) {
  const std::string edgelist1_filename = "edgelist1.txt";
  write_test_file(edgelist1_filename,
    "A\tB\n"
    "A\tC\n"
    "B\tC\n"
    "C\tD\n"
  );

  const std::string edgelist2_filename = "edgelist2.txt";
  write_test_file(edgelist2_filename,
    "A\tB\n"
    "B\tC\n"
    "B\tD\n"
    "C\tE\n"
  );

  const std::string filename = "test_flist.txt";
  write_test_file(filename,
    "edgelist1.txt\tlayer1\n"
    "edgelist2.txt\tlayer2\n"
  );

  MultiplexProtected mp(filename, false);

  auto local_layer_list = mp.create_local_layer_list({true, false});
  ASSERT_EQ(local_layer_list.size(), 2);
  EXPECT_EQ(local_layer_list[0], true);
  EXPECT_EQ(local_layer_list[1], false);

  std::remove(edgelist1_filename.c_str());
  std::remove(edgelist2_filename.c_str());
  std::remove(filename.c_str());
}

TEST(TestMultiplex, CreateLocalLabelListEmptyInput) {
  const std::string edgelist1_filename = "edgelist1.txt";
  write_test_file(edgelist1_filename,
    "A\tB\n"
    "A\tC\n"
    "B\tC\n"
    "C\tD\n"
  );

  const std::string edgelist2_filename = "edgelist2.txt";
  write_test_file(edgelist2_filename,
    "A\tB\n"
    "B\tC\n"
    "B\tD\n"
    "C\tE\n"
  );

  const std::string filename = "test_flist.txt";
  write_test_file(filename,
    "edgelist1.txt\tlayer1\n"
    "edgelist2.txt\tlayer2\n"
  );

  MultiplexProtected mp(filename, false);

  const auto local_label_list = mp.create_local_label_list({});

  ASSERT_EQ(local_label_list.size(), 5);
  EXPECT_EQ(local_label_list[0], "A");
  EXPECT_EQ(local_label_list[1], "B");
  EXPECT_EQ(local_label_list[2], "C");
  EXPECT_EQ(local_label_list[3], "D");
  EXPECT_EQ(local_label_list[4], "E");

  std::remove(edgelist1_filename.c_str());
  std::remove(edgelist2_filename.c_str());
  std::remove(filename.c_str());
}

TEST(TestMultiplex, CreateLocalLabelListNonemptyInput) {
  const std::string edgelist1_filename = "edgelist1.txt";
  write_test_file(edgelist1_filename,
    "A\tB\n"
    "A\tC\n"
    "B\tC\n"
    "C\tD\n"
  );

  const std::string edgelist2_filename = "edgelist2.txt";
  write_test_file(edgelist2_filename,
    "A\tB\n"
    "B\tC\n"
    "B\tD\n"
    "C\tE\n"
  );

  const std::string filename = "test_flist.txt";
  write_test_file(filename,
    "edgelist1.txt\tlayer1\n"
    "edgelist2.txt\tlayer2\n"
  );

  MultiplexProtected mp(filename, false);

  const auto local_label_list = mp.create_local_label_list({"A","C","E","D","F","G"});

  ASSERT_EQ(local_label_list.size(), 6);
  EXPECT_EQ(local_label_list[0], "A");
  EXPECT_EQ(local_label_list[1], "C");
  EXPECT_EQ(local_label_list[2], "E");
  EXPECT_EQ(local_label_list[3], "D");
  EXPECT_EQ(local_label_list[4], "F");
  EXPECT_EQ(local_label_list[5], "G");

  std::remove(edgelist1_filename.c_str());
  std::remove(edgelist2_filename.c_str());
  std::remove(filename.c_str());
}

TEST(TestMultiplex, GetIntraLayerTransitionMatrixThrowOnSmallDelta) {
  const std::string edgelist1_filename = "edgelist1.txt";
  write_test_file(edgelist1_filename,
    "A\tB\n"
    "A\tC\n"
    "B\tC\n"
    "C\tD\n"
  );

  const std::string edgelist2_filename = "edgelist2.txt";
  write_test_file(edgelist2_filename,
    "A\tB\n"
    "B\tC\n"
    "B\tD\n"
    "C\tE\n"
  );

  const std::string filename = "test_flist.txt";
  write_test_file(filename,
    "edgelist1.txt\tlayer1\n"
    "edgelist2.txt\tlayer2\n"
  );

  Multiplex mp(filename);

  ASSERT_THAT(
    [&](){mp.get_intra_layer_transition_matrix(-1); },
    testing::ThrowsMessage<std::invalid_argument>("Multiplex::get_intra_layer_transition_matrix - delta must be between 0.0 and 1.0")
  );

  std::remove(edgelist1_filename.c_str());
  std::remove(edgelist2_filename.c_str());
  std::remove(filename.c_str());
}

TEST(TestMultiplex, GetIntraLayerTransitionMatrixThrowOnBigDelta) {
  const std::string edgelist1_filename = "edgelist1.txt";
  write_test_file(edgelist1_filename,
    "A\tB\n"
    "A\tC\n"
    "B\tC\n"
    "C\tD\n"
  );

  const std::string edgelist2_filename = "edgelist2.txt";
  write_test_file(edgelist2_filename,
    "A\tB\n"
    "B\tC\n"
    "B\tD\n"
    "C\tE\n"
  );

  const std::string filename = "test_flist.txt";
  write_test_file(filename,
    "edgelist1.txt\tlayer1\n"
    "edgelist2.txt\tlayer2\n"
  );

  Multiplex mp(filename);

  ASSERT_THAT(
    [&](){mp.get_intra_layer_transition_matrix(2.0); },
    testing::ThrowsMessage<std::invalid_argument>("Multiplex::get_intra_layer_transition_matrix - delta must be between 0.0 and 1.0")
  );

  std::remove(edgelist1_filename.c_str());
  std::remove(edgelist2_filename.c_str());
  std::remove(filename.c_str());
}

TEST(TestMultiplex, CorrectlyCreatesIntraTranMatrix) {
  const std::string edgelist1_filename = "edgelist1.txt";
  write_test_file(edgelist1_filename,
    "A\tB\n"
    "A\tC\n"
    "B\tC\n"
    "C\tD\n"
  );

  const std::string edgelist2_filename = "edgelist2.txt";
  write_test_file(edgelist2_filename,
    "A\tB\n"
    "B\tC\n"
    "B\tD\n"
    "C\tE\n"
  );

  const std::string filename = "test_flist.txt";
  write_test_file(filename,
    "edgelist1.txt\tlayer1\n"
    "edgelist2.txt\tlayer2\n"
  );

  Multiplex mp(filename);
  CSR_Matrix intra_tran_mat = mp.get_intra_layer_transition_matrix(0.6);

  ASSERT_EQ(intra_tran_mat.n_rows(), 10);
  ASSERT_EQ(intra_tran_mat.n_cols(), 10);
  ASSERT_EQ(intra_tran_mat.nnz(), 16);

  const auto values = intra_tran_mat.get_values();
  const auto col_idx = intra_tran_mat.get_col_idx();
  const auto row_ptr = intra_tran_mat.get_row_ptr();

  ASSERT_EQ(values.size(), 16);
  ASSERT_EQ(col_idx.size(), 16);
  ASSERT_EQ(row_ptr.size(),11);

  EXPECT_NEAR(values[0], 0.4/2.0, 1e-8);
  EXPECT_NEAR(values[1], 0.4/3.0, 1e-8);
  EXPECT_NEAR(values[2], 0.4/2.0, 1e-8);
  EXPECT_NEAR(values[3], 0.4/3.0, 1e-8);
  EXPECT_NEAR(values[4], 0.4/2.0, 1e-8);
  EXPECT_NEAR(values[5], 0.4/2.0, 1e-8);
  EXPECT_NEAR(values[6], 0.4/1.0, 1e-8);
  EXPECT_NEAR(values[7], 0.4/3.0, 1e-8);

  EXPECT_NEAR(values[8], 0.4/3.0, 1e-8);
  EXPECT_NEAR(values[9], 0.4/1.0, 1e-8);
  EXPECT_NEAR(values[10], 0.4/2.0, 1e-8);
  EXPECT_NEAR(values[11], 0.4/1.0, 1e-8);
  EXPECT_NEAR(values[12], 0.4/3.0, 1e-8);
  EXPECT_NEAR(values[13], 0.4/1.0, 1e-8);
  EXPECT_NEAR(values[14], 0.4/3.0, 1e-8);
  EXPECT_NEAR(values[15], 0.4/2.0, 1e-8);

  EXPECT_EQ(col_idx[0], 1);
  EXPECT_EQ(col_idx[1], 2);
  EXPECT_EQ(col_idx[2], 0);
  EXPECT_EQ(col_idx[3], 2);
  EXPECT_EQ(col_idx[4], 0);
  EXPECT_EQ(col_idx[5], 1);
  EXPECT_EQ(col_idx[6], 3);
  EXPECT_EQ(col_idx[7], 2);

  EXPECT_EQ(col_idx[8], 6);
  EXPECT_EQ(col_idx[9], 5);
  EXPECT_EQ(col_idx[10], 7);
  EXPECT_EQ(col_idx[11], 8);
  EXPECT_EQ(col_idx[12], 6);
  EXPECT_EQ(col_idx[13], 9);
  EXPECT_EQ(col_idx[14], 6);
  EXPECT_EQ(col_idx[15], 7);

  EXPECT_EQ(row_ptr[0], 0);
  EXPECT_EQ(row_ptr[1], 2);
  EXPECT_EQ(row_ptr[2], 4);
  EXPECT_EQ(row_ptr[3], 7);
  EXPECT_EQ(row_ptr[4], 8);
  EXPECT_EQ(row_ptr[5], 8);
  EXPECT_EQ(row_ptr[6], 9);
  EXPECT_EQ(row_ptr[7], 12);
  EXPECT_EQ(row_ptr[8], 14);
  EXPECT_EQ(row_ptr[9], 15);
  EXPECT_EQ(row_ptr[10], 16);

  std::remove(edgelist1_filename.c_str());
  std::remove(edgelist2_filename.c_str());
  std::remove(filename.c_str());
}

TEST(TestMultiplex, CorrectlyCreatesIntraTranMatrixWithSkippedLayer) {
  const std::string edgelist1_filename = "edgelist1.txt";
  write_test_file(edgelist1_filename,
    "A\tB\n"
    "A\tC\n"
    "B\tC\n"
    "C\tD\n"
  );

  const std::string edgelist2_filename = "edgelist2.txt";
  write_test_file(edgelist2_filename,
    "A\tB\n"
    "B\tC\n"
    "B\tD\n"
    "C\tE\n"
  );

  const std::string filename = "test_flist.txt";
  write_test_file(filename,
    "edgelist1.txt\tlayer1\n"
    "edgelist2.txt\tlayer2\n"
  );

  Multiplex mp(filename);
  CSR_Matrix intra_tran_mat = mp.get_intra_layer_transition_matrix(0.6, {}, {true, false});

  ASSERT_EQ(intra_tran_mat.n_rows(), 5);
  ASSERT_EQ(intra_tran_mat.n_cols(), 5);
  ASSERT_EQ(intra_tran_mat.nnz(), 8);

  const auto values = intra_tran_mat.get_values();
  const auto col_idx = intra_tran_mat.get_col_idx();
  const auto row_ptr = intra_tran_mat.get_row_ptr();

  ASSERT_EQ(values.size(), 8);
  ASSERT_EQ(col_idx.size(), 8);
  ASSERT_EQ(row_ptr.size(),6);

  EXPECT_NEAR(values[0], 0.4/2.0, 1e-8);
  EXPECT_NEAR(values[1], 0.4/3.0, 1e-8);
  EXPECT_NEAR(values[2], 0.4/2.0, 1e-8);
  EXPECT_NEAR(values[3], 0.4/3.0, 1e-8);
  EXPECT_NEAR(values[4], 0.4/2.0, 1e-8);
  EXPECT_NEAR(values[5], 0.4/2.0, 1e-8);
  EXPECT_NEAR(values[6], 0.4/1.0, 1e-8);
  EXPECT_NEAR(values[7], 0.4/3.0, 1e-8);

  EXPECT_EQ(col_idx[0], 1);
  EXPECT_EQ(col_idx[1], 2);
  EXPECT_EQ(col_idx[2], 0);
  EXPECT_EQ(col_idx[3], 2);
  EXPECT_EQ(col_idx[4], 0);
  EXPECT_EQ(col_idx[5], 1);
  EXPECT_EQ(col_idx[6], 3);
  EXPECT_EQ(col_idx[7], 2);

  EXPECT_EQ(row_ptr[0], 0);
  EXPECT_EQ(row_ptr[1], 2);
  EXPECT_EQ(row_ptr[2], 4);
  EXPECT_EQ(row_ptr[3], 7);
  EXPECT_EQ(row_ptr[4], 8);
  EXPECT_EQ(row_ptr[5], 8);

  std::remove(edgelist1_filename.c_str());
  std::remove(edgelist2_filename.c_str());
  std::remove(filename.c_str());
}

TEST(TestMultiplex, GetInterLayerTransitionMatrixThrowOnSmallDelta) {
  Multiplex mp;
  CSR_Matrix intra_tran;

  ASSERT_THAT(
    [&](){mp.get_inter_layer_transition_matrix(intra_tran, -1.0); },
    testing::ThrowsMessage<std::invalid_argument>("Multiplex::get_inter_layer_transition_matrix - delta must be between 0.0 and 1.0")
  );
}

TEST(TestMultiplex, GetInterLayerTransitionMatrixThrowOnBigDelta) {
  Multiplex mp;
  CSR_Matrix intra_tran;

  ASSERT_THAT(
    [&](){mp.get_inter_layer_transition_matrix(intra_tran, 2.0); },
    testing::ThrowsMessage<std::invalid_argument>("Multiplex::get_inter_layer_transition_matrix - delta must be between 0.0 and 1.0")
  );
}

TEST(TestMultiplex, GetIntraLayerTransitionMatrixReturnsEmptyVectorWhenL_IsZero) {
  const std::string edgelist1_filename = "edgelist1.txt";
  write_test_file(edgelist1_filename,
  "A\tB\n"
  "A\tC\n"
  "B\tC\n"
  "C\tD\n"
  );

  const std::string edgelist2_filename = "edgelist2.txt";
  write_test_file(edgelist2_filename,
  "A\tB\n"
  "B\tC\n"
  "B\tD\n"
  "C\tE\n"
  );

  const std::string filename = "test_flist.txt";
  write_test_file(filename,
  "edgelist1.txt\tlayer1\n"
  "edgelist2.txt\tlayer2\n"
  );

  Multiplex mp(filename);
  CSR_Matrix intra_tran;

  std::vector<bool> layer_list = {false, false};

  auto inter_tran = mp.get_inter_layer_transition_matrix(intra_tran, 0.5, {}, layer_list);

  EXPECT_EQ(inter_tran.size(), 0);

  std::remove(edgelist1_filename.c_str());
  std::remove(edgelist2_filename.c_str());
  std::remove(filename.c_str());
}

TEST(TestMultiplex, GetIntraLayerTransitionMatrixReturnsVectorOfAllZerosWhenL_IsOne) {
  const std::string edgelist1_filename = "edgelist1.txt";
  write_test_file(edgelist1_filename,
    "A\tB\n"
    "A\tC\n"
    "B\tC\n"
    "C\tD\n"
  );

  const std::string edgelist2_filename = "edgelist2.txt";
  write_test_file(edgelist2_filename,
   "A\tB\n"
   "B\tC\n"
   "B\tD\n"
   "C\tE\n"
  );

  const std::string filename = "test_flist.txt";
  write_test_file(filename,
   "edgelist1.txt\tlayer1\n"
   "edgelist2.txt\tlayer2\n"
  );

  Multiplex mp(filename);
  const std::size_t N = mp.n_nodes();
  CSR_Matrix intra_tran;

  std::vector<bool> layer_list = {true, false};

  auto inter_tran = mp.get_inter_layer_transition_matrix(intra_tran, 0.5, {}, layer_list);

  std::vector<double> expected_inter_tran(N, 0.0);
  EXPECT_EQ(inter_tran, expected_inter_tran);
  
  std::remove(edgelist1_filename.c_str());
  std::remove(edgelist2_filename.c_str());
  std::remove(filename.c_str());
}

TEST(TestMultiplex, GetIntraLayerTransitionMatrixReturnsCorrectValuesForL_GreaterThanOne) {
  const std::string edgelist1_filename = "edgelist1.txt";
  write_test_file(edgelist1_filename,
    "A\tB\n"
    "A\tC\n"
    "B\tC\n"
    "C\tD\n"
  );

  const std::string edgelist2_filename = "edgelist2.txt";
  write_test_file(edgelist2_filename,
    "A\tB\n"
    "B\tC\n"
    "B\tD\n"
    "C\tE\n"
  );

  const std::string edgelist3_filename = "edgelist3.txt";
  write_test_file(edgelist3_filename,
    "B\tC\n"
    "C\tE\n"
  );

  const std::string filename = "test_flist.txt";
  write_test_file(filename,
    "edgelist1.txt\tlayer1\n"
    "edgelist2.txt\tlayer2\n"
    "edgelist3.txt\tlayer3\n"
  );

  Multiplex mp(filename);
  CSR_Matrix intra_tran = mp.get_intra_layer_transition_matrix();

  auto inter_tran = mp.get_inter_layer_transition_matrix(intra_tran);

  std::vector<double> expected_inter_tran = {0.25, 0.25, 0.25, 0.25, 0.5,
                                             0.25, 0.25, 0.25, 0.25, 0.25,
                                             0.5, 0.25, 0.25, 0.5, 0.25};
  EXPECT_EQ(inter_tran, expected_inter_tran);
  
  std::remove(edgelist1_filename.c_str());
  std::remove(edgelist2_filename.c_str());
  std::remove(edgelist3_filename.c_str());
  std::remove(filename.c_str());
}

TEST(TestMultiplex, GetNodesByLayerReturnsCorrectVales) {
  const std::string edgelist1_filename = "edgelist1.txt";
  write_test_file(edgelist1_filename,
    "A\tB\n"
    "A\tD\n"
    "B\tD\n"
  );

  const std::string edgelist2_filename = "edgelist2.txt";
  write_test_file(edgelist2_filename,
    "A\tB\n"
    "B\tC\n"
    "B\tD\n"
    "C\tE\n"
  );

  const std::string edgelist3_filename = "edgelist3.txt";
  write_test_file(edgelist3_filename,
    "B\tC\n"
    "B\tF\n"
    "C\tD\n"
    "C\tE\n"
    "E\tF\n"
    "E\tG\n"
  );

  const std::string filename = "test_flist.txt";
  write_test_file(filename,
    "edgelist1.txt\tlayer1\n"
    "edgelist2.txt\tlayer2\n"
    "edgelist3.txt\tlayer3\n"
  );

  Multiplex mp(filename);
  std::vector<std::string> nodes_in_mp = mp.get_nodes();
  std::vector<bool> nodes_to_layer = mp.get_nodes_by_layer(nodes_in_mp);

  std::remove(edgelist1_filename.c_str());
  std::remove(edgelist2_filename.c_str());
  std::remove(edgelist3_filename.c_str());
  std::remove(filename.c_str());

  ASSERT_EQ(nodes_to_layer.size(), 21);

  EXPECT_TRUE(nodes_to_layer[0]);   // A
  EXPECT_TRUE(nodes_to_layer[1]);
  EXPECT_FALSE(nodes_to_layer[2]);
  EXPECT_TRUE(nodes_to_layer[3]);   // B
  EXPECT_TRUE(nodes_to_layer[4]);
  EXPECT_TRUE(nodes_to_layer[5]);
  EXPECT_FALSE(nodes_to_layer[6]);  // C
  EXPECT_TRUE(nodes_to_layer[7]);
  EXPECT_TRUE(nodes_to_layer[8]);
  EXPECT_TRUE(nodes_to_layer[9]);   // D
  EXPECT_TRUE(nodes_to_layer[10]);
  EXPECT_TRUE(nodes_to_layer[11]);
  EXPECT_FALSE(nodes_to_layer[12]);  // E
  EXPECT_TRUE(nodes_to_layer[13]);
  EXPECT_TRUE(nodes_to_layer[14]);
  EXPECT_FALSE(nodes_to_layer[15]);  // F
  EXPECT_FALSE(nodes_to_layer[16]);
  EXPECT_TRUE(nodes_to_layer[17]);
  EXPECT_FALSE(nodes_to_layer[18]);  // G
  EXPECT_FALSE(nodes_to_layer[19]);
  EXPECT_TRUE(nodes_to_layer[20]);
}

TEST(TestMultiplex, MergeLayers) {
    const std::string edgelist1_filename = "edgelist1.txt";
  write_test_file(edgelist1_filename,
    "A\tB\t1.0\n"
    "A\tC\t0.8\n"
    "B\tC\t0.7\n"
    "C\tD\t1.0\n"
  );

  const std::string edgelist2_filename = "edgelist2.txt";
  write_test_file(edgelist2_filename,
    "A\tB\t0.2\n"
    "B\tC\t0.84\n"
    "B\tD\t0.6\n"
    "C\tE\t0.5\n"
  );

  const std::string filename = "test_flist.txt";
  write_test_file(filename,
    "edgelist1.txt\tlayer1\n"
    "edgelist2.txt\tlayer2\n"
  );

  MultiplexProtected mp(filename, false);


  Network merged = mp.merge_layers(MergeMethod::Max);

  std::vector<std::string> expected_labels = {"A","B","C","D","E"};
  auto actual_labels = merged.get_labels();
  EXPECT_EQ(actual_labels, expected_labels);

  EXPECT_EQ(6, merged.get_n_edges());

  EXPECT_DOUBLE_EQ(merged.get_edge_weight("A", "B"), 1.0);
  EXPECT_DOUBLE_EQ(merged.get_edge_weight("B", "A"), 1.0);

  EXPECT_DOUBLE_EQ(merged.get_edge_weight("A", "C"), 0.8);
  EXPECT_DOUBLE_EQ(merged.get_edge_weight("C", "A"), 0.8);

  EXPECT_DOUBLE_EQ(merged.get_edge_weight("B", "C"), 0.84);
  EXPECT_DOUBLE_EQ(merged.get_edge_weight("C", "B"), 0.84);

  EXPECT_DOUBLE_EQ(merged.get_edge_weight("B", "D"), 0.6);
  EXPECT_DOUBLE_EQ(merged.get_edge_weight("D", "B"), 0.6);

  EXPECT_DOUBLE_EQ(merged.get_edge_weight("C", "D"), 1.0);
  EXPECT_DOUBLE_EQ(merged.get_edge_weight("D", "C"), 1.0);

  EXPECT_DOUBLE_EQ(merged.get_edge_weight("C", "E"), 0.5);
  EXPECT_DOUBLE_EQ(merged.get_edge_weight("E", "C"), 0.5);

  std::remove(edgelist1_filename.c_str());
  std::remove(edgelist2_filename.c_str());
  std::remove(filename.c_str());
}
