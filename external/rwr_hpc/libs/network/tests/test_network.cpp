#include <gtest/gtest.h>
#include <gmock/gmock-matchers.h>
#include <gmock/gmock.h>

#include <fstream>
#include <cmath>
#include <exception>
#include <string>
#include <algorithm>

#include <network/Network.hpp>

class NetworkMockAddNode : public Network {
  public:
    MOCK_METHOD(std::size_t, add_node, (const std::string &label), (override));
};

class NetworkMockAddEdge : public Network {
  public:
    MOCK_METHOD(uint32_t, get_idx_of_label, (const std::string &label), (const, override));
    MOCK_METHOD(
      void,
      add_edge_core,
      (const uint32_t src_idx, const uint32_t tgt_idx, const double weight),
      (override)
    );
    MOCK_METHOD(
      bool,
      update_edge,
      (const uint32_t src_idx, const uint32_t tgt_idx, const double weight),
      (override)
    );
    MOCK_METHOD(
      std::size_t,
      add_node,
      (const std::string& label),
      (override)
    );
    MOCK_METHOD(
      std::size_t,
      degree,
      (const std::size_t idx),
      (const, override)
    );
};

class NetworkMockGetAdjMatrix : public Network {
  public:
    MOCK_METHOD(
      CSR_Matrix,
      get_adjacency_matrix,
      (const std::vector<std::string> &label_list),
      (const, override)
    );
};

class NetworkMockShortestPathSingle : public Network {
  public:
      MOCK_METHOD(std::vector<std::vector<uint32_t>>, 
                find_all_shortest_paths_bfs_core,
                (uint32_t s_idx, const std::unordered_set<uint32_t>& target_indices),
                (const, override));
    
    MOCK_METHOD(std::vector<std::vector<uint32_t>>, 
                find_all_shortest_paths_dijkstra_core,
                (uint32_t s_idx, const std::unordered_set<uint32_t>& target_indices),
                (const, override));

    MOCK_METHOD(std::vector<std::vector<std::string>>, 
                convert_paths_to_labels,
                (const std::vector<std::vector<uint32_t>>& index_paths),
                (const, override));
};

class NetworkMockShortestPath : public Network {
  public:
    MOCK_METHOD(std::vector<std::vector<std::string>>, 
                find_all_shortest_paths_bfs,
                (const std::unordered_set<std::string>& S, const std::unordered_set<std::string>& T),
                (const, override));
    
    MOCK_METHOD(std::vector<std::vector<std::string>>, 
                find_all_shortest_paths_dijkstra,
                (const std::unordered_set<std::string>& S, const std::unordered_set<std::string>& T),
                (const, override));
};

class NetworkMockSingleToMulti : public Network {
  public:
    MOCK_METHOD(std::vector<std::vector<std::string>>, 
                find_all_shortest_paths,
                (const std::unordered_set<std::string>& S, const std::unordered_set<std::string>& T, const bool use_weights),
                (const, override));

};

class NetworkProtected : public Network {
  public: 
    using Network::add_edge_core;
    using Network::update_edge;
    using Network::create_local_label_list;
    using Network::get_transition_matrix_size;
    using Network::merge_method_from_string;
    using Network::merge_method_to_string;
    using Network::pack_edge;
    using Network::convert_paths_to_labels;
    using Network::reconstruct_paths;
    using Network::find_all_shortest_paths_bfs_core;
    using Network::find_all_shortest_paths_dijkstra_core;
};

// Helper to create temporary test file
void write_test_file(const std::string& filename, const std::string& content) {
  std::ofstream out(filename);
  out << content;
  out.close();
}

TEST(TestNetwork, GetN_NodesReturnsZeroOnEmptyNetwork) {
  Network net;

  EXPECT_EQ(net.get_n_nodes(), 0);
}

TEST(TestNetwork, GetN_NodesReturnsCorrectNumberOfNodes) {
  Network net;

  net.add_node("A");
  net.add_node("C");
  net.add_node("B");
  net.add_node("A");

  EXPECT_EQ(net.get_n_nodes(), 3);
}

TEST(TestNetwork, GetLabelsReturnsEmptyStringWhenNoNodesInNetwork) {
  Network net;

  auto actual_labels = net.get_labels();

  EXPECT_TRUE(actual_labels.empty());
}

TEST(TestNetwork, GetLabelsReturnsLabelsInOrder) {
  Network net;
  std::vector<std::string> expected_labels = {"A", "B", "C"};
  net.add_node("A");
  net.add_node("C");
  net.add_node("B");

  auto actual_labels = net.get_labels();
  EXPECT_EQ(actual_labels, expected_labels);
}

TEST(TestNetwork, GetIdxOfLabelThrowsIfLabelNotInNetwork) {
  Network net;
  ASSERT_THAT(
    [&](){auto result = net.get_idx_of_label("A"); },
    testing::ThrowsMessage<std::runtime_error>("Network::get_idx_of_label - could not find label in labels_")
  );

  net.add_node("A");

  ASSERT_THAT(
    [&](){auto result = net.get_idx_of_label("B"); },
    testing::ThrowsMessage<std::runtime_error>("Network::get_idx_of_label - could not find label in labels_")
  );
}

TEST(TestNetwork, GetIdxOfLabelReturnsCorrectIndex) {
  Network net;
  net.add_node("A");
  net.add_node("C");
  net.add_node("B");

  EXPECT_EQ(net.get_idx_of_label("A"), 0);
  EXPECT_EQ(net.get_idx_of_label("C"), 1);
  EXPECT_EQ(net.get_idx_of_label("B"), 2);
}

TEST(TestNetwork, GetLableOfIdxThrowsInIdxIsOutOfRange) {
  Network net;
  net.add_nodes({"A","B","C","D"});

  EXPECT_THROW(net.get_label_of_index(5), std::out_of_range);
}

TEST(TestNetwork, GetLabelOfIdxReturnsCorrectLabel) {
  Network net;
  net.add_nodes({"B","C","A","D"});

  EXPECT_EQ(net.get_label_of_index(2), "A");
  EXPECT_EQ(net.get_label_of_index(0), "B");
  EXPECT_EQ(net.get_label_of_index(1), "C");
  EXPECT_EQ(net.get_label_of_index(3), "D");
}

TEST(TestNetwork, GetN_EdgesHandlesNoEdgesUndirected) {
  Network net;
  net.set_directed(false);
  EXPECT_EQ(net.get_n_edges(), 0);
}

TEST(TestNetwork, GetN_EdgesHandlesNoEdgesDirected) {
  Network net;
  net.set_directed(true);
  EXPECT_EQ(net.get_n_edges(), 0);
}

TEST(TestNetwork, GetN_EdgesReturnsCorrectNumberUndirected) {
  Network net;
  net.set_directed(false);
  net.add_nodes({"A","B","C","D"});
  net.add_edge("A", "B", 0.9);
  net.add_edge("A", "C", 0.8);
  net.add_edge("B", "C", 0.7);
  net.add_edge("C", "D");

  EXPECT_EQ(net.get_n_edges(), 4);
}

TEST(TestNetwork, GetN_EdgesReturnsCorrectNumberDirected) {
  Network net;
  net.set_directed(true);
  net.add_nodes({"A","B","C","D"});
  net.add_edge("A", "B", 0.9);
  net.add_edge("A", "C", 0.8);
  net.add_edge("B", "A", 1.0);
  net.add_edge("B", "C", 0.7);
  net.add_edge("C", "D");

  EXPECT_EQ(net.get_n_edges(), 5);
}

TEST(TestNetwork, GetNnzReturnZeroWhenNoEdges) {
  Network net;
  EXPECT_EQ(net.get_nnz(), 0);

  net.add_nodes({"A","B","C"});
  EXPECT_EQ(net.get_nnz(), 0);
}

TEST(TestNetwork, GetNnzReturnsCorrectValueOnDefaultInput) {
  Network net;
  net.add_nodes({"A","B","C","D"});
  net.add_edge("A","B");
  net.add_edge("A","C");
  net.add_edge("B","C");
  net.add_edge("C","D");

  std::size_t expected_nnz = 8;
  std::size_t actual_nnz = net.get_nnz();

  EXPECT_EQ(actual_nnz, expected_nnz);
}

TEST(TestNetwork, GetNnzReturnsCorrectValueOnSubsetOfNodes) {
  Network net;
  net.add_nodes({"A","B","C","D"});
  net.add_edge("A","B");
  net.add_edge("A","C");
  net.add_edge("B","C");
  net.add_edge("C","D");
  std::vector<std::string> label_list = {"A","B", "D"};

  std::size_t expected_nnz = 2;
  std::size_t actual_nnz = net.get_nnz(label_list);

  EXPECT_EQ(actual_nnz, expected_nnz);
}

TEST(TestNetwork, GetNnzReturnsCorrectValueOnSupersetOfNodes) {
  Network net;
  net.add_nodes({"A","B","C","D"});
  net.add_edge("A","B");
  net.add_edge("A","C");
  net.add_edge("B","C");
  net.add_edge("C","D");
  std::vector<std::string> label_list = {"A","B","C","D","E"};

  std::size_t expected_nnz = 8;
  std::size_t actual_nnz = net.get_nnz(label_list);

  EXPECT_EQ(actual_nnz, expected_nnz);
}

TEST(TestNetwork, GetNnzReturnsCorrectValueDefaultForDirectedNetwork) {
  Network net;
  net.set_directed(true);
  net.add_nodes({"A","B","C","D"});
  net.add_edge("A","B");
  net.add_edge("A","C");
  net.add_edge("B","C");
  net.add_edge("C","D");

  std::size_t expected_nnz = 4;
  std::size_t actual_nnz = net.get_nnz();

  EXPECT_EQ(actual_nnz, expected_nnz);
}

TEST(TestNetwork, DegreeThrowsWhenIdxIsOutOfRange) {
  Network net;
  net.add_nodes({"A","B","C","D"});
  net.add_edge("A","B");
  net.add_edge("A","C");
  net.add_edge("A","D");
  net.add_edge("B","C");

  ASSERT_THAT(
    [&](){net.degree(10); },
    testing::ThrowsMessage<std::out_of_range>("Network::degree - trying to access vertex index outside of range")
  );
}

TEST(TestNetwork, DegreeThrowsForDirectedNetwork) {
  Network net;
  net.set_directed(true);
  net.add_nodes({"A","B","C","D"});
  net.add_edge("A","B");
  net.add_edge("A","C");
  net.add_edge("A","D");
  net.add_edge("B","C");

  ASSERT_THAT(
    [&](){net.degree(1); },
    testing::ThrowsMessage<std::runtime_error>("Network::degree - not implemented for directed networks")
  );
}

TEST(TestNetwork, DegreeReturnsCorrectValue) {
  Network net;
  net.add_nodes({"A","B","C","D"});
  net.add_edge("A","B");
  net.add_edge("A","C");
  net.add_edge("A","D");
  net.add_edge("B","C");

  EXPECT_EQ(net.degree(0), 3UL);
  EXPECT_EQ(net.degree(1), 2UL);
  EXPECT_EQ(net.degree(2), 2UL);
  EXPECT_EQ(net.degree(3), 1UL);
}

TEST(TestNetwork, GetMaxDegreeReturnsZeroWhenNoEdges) {
  Network net;
  net.add_nodes({"A","B","C","D","E"});

  EXPECT_EQ(net.get_max_degree(), 0);
}

TEST(TestNetwork, GetMaxDegreeReturnsCorrectValue) {
    Network net;
  net.add_nodes({"A","B","C","D"});
  net.add_edge("A","B");
  net.add_edge("A","C");
  net.add_edge("A","D");
  net.add_edge("B","C");

  EXPECT_EQ(net.get_max_degree(), 3UL);
}

TEST(TestNetwork, ContainsNodeReturnsFalseForEmptyNetwwork) {
  Network net;

  EXPECT_FALSE(net.contains_node("A"));
}

TEST(TestNetwork, ContainsNodeReturnsTrueForExistingNode) {
  Network net;
  net.add_nodes({"A","B","C","D"});

  EXPECT_TRUE(net.contains_node("A"));
}

TEST(TestNetwork, ContainsNodeReturnsFalseForNonexistingNode) {
  Network net;
  net.add_nodes({"A","B","C","D"});

  EXPECT_FALSE(net.contains_node("E"));
}

TEST(TestNetwork, AddNodeCorrectlyAddsNodeToEmptyNetwork) {
  Network net;
  EXPECT_EQ(net.get_n_nodes(), 0);

  net.add_node("A");

  EXPECT_EQ(net.get_n_nodes(), 1);
  auto actual_lables = net.get_labels();
  std::vector<std::string> expected_lables = {"A"};

  EXPECT_EQ(actual_lables, expected_lables);
}

TEST(TestNetwork, AddNodeCorrectlyAddsNodeToNonemptyNetwork) {
  Network net;
  net.add_node("A");
  net.add_node("B");

  EXPECT_EQ(net.get_n_nodes(), 2);
  auto actual_lables = net.get_labels();
  std::vector<std::string> expected_lables = {"A","B"};

  EXPECT_EQ(actual_lables, expected_lables);
}

TEST(TestNetwork, AddNodeCorrectlyHandlesExistingNodeLabel) {
  Network net;
  net.add_node("A");
  net.add_node("A");

  EXPECT_EQ(net.get_n_nodes(), 1);
  auto actual_lables = net.get_labels();
  std::vector<std::string> expected_lables = {"A"};

  EXPECT_EQ(actual_lables, expected_lables);

  EXPECT_EQ(net.get_idx_of_label("A"), 0);
}

TEST(TestNetwork, AddNodesCallAddNodeCorrectNumberOfTimes) {
  NetworkMockAddNode net;

  EXPECT_CALL(net, add_node("1")).Times(1);
  EXPECT_CALL(net, add_node("4")).Times(1);
  EXPECT_CALL(net, add_node("2")).Times(1);

  net.add_nodes({"1", "4", "2"});
}

TEST(TestNetwork, AddEdgeSkipsIfWeightIsZero) {
  NetworkMockAddEdge mock_net;

  EXPECT_CALL(mock_net, get_idx_of_label(::testing::_))
    .Times(0);
  EXPECT_CALL(mock_net, add_node(::testing::_))
    .Times(0);
  EXPECT_CALL(mock_net, add_edge_core(::testing::_, ::testing::_, ::testing::_))
    .Times(0);
  EXPECT_CALL(mock_net, update_edge(::testing::_,::testing::_,::testing::_))
    .Times(0);
  EXPECT_CALL(mock_net, degree(::testing::_))
    .Times(0);

  EXPECT_FALSE(mock_net.add_edge("1","2", 0.0));
}

TEST(TestNetwork, AddEdgeThrowsIfSrcMissingAndMissingNotAllowed) {
  NetworkMockAddEdge mock_net;

  EXPECT_CALL(mock_net, get_idx_of_label("4"))
    .WillOnce(::testing::Throw(std::runtime_error("Simulated error")));
  EXPECT_CALL(mock_net, add_node(::testing::_))
    .Times(0);
  EXPECT_CALL(mock_net, add_edge_core(::testing::_,::testing::_,::testing::_))
    .Times(0);
  EXPECT_CALL(mock_net, update_edge(::testing::_,::testing::_,::testing::_))
    .Times(0);
  EXPECT_CALL(mock_net, degree(::testing::_))
    .Times(0);

  ASSERT_THAT(
    [&](){mock_net.add_edge("4","2", 1.0); },
    testing::ThrowsMessage<std::runtime_error>("Simulated error")
  );
}

TEST(TestNetwork, AddEdgeThrowsIfTgtMissingAndMissingNotAllowed) {
  NetworkMockAddEdge mock_net;

  EXPECT_CALL(mock_net, get_idx_of_label("1"))
    .WillOnce(::testing::Return(0));
  EXPECT_CALL(mock_net, get_idx_of_label("4"))
    .WillOnce(::testing::Throw(std::runtime_error("Simulated error")));
  EXPECT_CALL(mock_net, add_node(::testing::_))
    .Times(0);
  EXPECT_CALL(mock_net, add_edge_core(::testing::_,::testing::_,::testing::_))
    .Times(0);
  EXPECT_CALL(mock_net, update_edge(::testing::_,::testing::_,::testing::_))
    .Times(0);
  EXPECT_CALL(mock_net, degree(::testing::_))
    .Times(0);

  ASSERT_THAT(
    [&](){mock_net.add_edge("1","4", 1.0); },
    testing::ThrowsMessage<std::runtime_error>("Simulated error")
  );
}

TEST(TestNetwork, AddEdgeCallsAddnNodeIfNodeMissingAndAllowMissingIsSet) {
  NetworkMockAddEdge mock_net;

  EXPECT_CALL(mock_net, get_idx_of_label("4"))
    .WillOnce(::testing::Throw(std::runtime_error("Simulated error")));
  EXPECT_CALL(mock_net, get_idx_of_label("5"))
    .WillOnce(::testing::Throw(std::runtime_error("Simulated error")));
  
  EXPECT_CALL(mock_net, add_node("4"))
    .WillOnce(::testing::Return(3));
  EXPECT_CALL(mock_net, add_node("5"))
    .WillOnce(::testing::Return(4));

  EXPECT_CALL(mock_net, update_edge(3, 4, 1.0))
    .WillOnce(::testing::Return(false));
  EXPECT_CALL(mock_net, update_edge(4, 3, 1.0))
    .WillOnce(::testing::Return(false));

  EXPECT_CALL(mock_net, add_edge_core(3, 4, 1.0))
    .Times(1);
  EXPECT_CALL(mock_net, add_edge_core(4, 3, 1.0))
    .Times(1);
 
  
  EXPECT_CALL(mock_net, degree(3))
    .WillOnce(::testing::Return(1));
  EXPECT_CALL(mock_net, degree(4))
    .WillOnce(::testing::Return(3));

  EXPECT_TRUE(mock_net.add_edge("4","5", 1.0, true));
}

TEST(TestNetwork, AddEdgeMultiGraphDirectedNoUpdate) {
  NetworkMockAddEdge mock_net;
  mock_net.set_multigraph(true);
  mock_net.set_directed(true);

  EXPECT_CALL(mock_net, get_idx_of_label(::testing::_))
    .WillOnce(::testing::Return(0))
    .WillOnce(::testing::Return(1));
  EXPECT_CALL(mock_net, add_node(::testing::_))
    .Times(0);
  EXPECT_CALL(mock_net, add_edge_core(0, 1, 1.0))
    .Times(1);
  EXPECT_CALL(mock_net, add_edge_core(1, 0, 1.0))
    .Times(0);
  EXPECT_CALL(mock_net, update_edge(::testing::_,::testing::_,::testing::_))
    .Times(0);
  EXPECT_CALL(mock_net, degree(::testing::_))
    .Times(0);

  EXPECT_TRUE(mock_net.add_edge("1","2", 1.0));
}

TEST(TestNetwork, AddEdgeMultiGraphUndirectedNoUpdate) {
  NetworkMockAddEdge mock_net;
  mock_net.set_multigraph(true);
  mock_net.set_directed(false);

  EXPECT_CALL(mock_net, get_idx_of_label(::testing::_))
    .WillOnce(::testing::Return(0))
    .WillOnce(::testing::Return(1));
  EXPECT_CALL(mock_net, add_node(::testing::_))
    .Times(0);
  EXPECT_CALL(mock_net, add_edge_core(0, 1, 1.0))
    .Times(1);
  EXPECT_CALL(mock_net, add_edge_core(1, 0, 1.0))
    .Times(1);
  EXPECT_CALL(mock_net, update_edge(::testing::_, ::testing::_, ::testing::_))
    .Times(0);
  EXPECT_CALL(mock_net, degree(0))
    .Times(1);
  EXPECT_CALL(mock_net, degree(1))
    .Times(1);

  EXPECT_TRUE(mock_net.add_edge("1","2", 1.0));
}

TEST(TestNetwork, AddEdgeMonoGraphDirectedNoUpdate) {
  NetworkMockAddEdge mock_net;
  mock_net.set_multigraph(false);
  mock_net.set_directed(true);
  
  EXPECT_CALL(mock_net, get_idx_of_label(::testing::_))
    .WillOnce(::testing::Return(0))
    .WillOnce(::testing::Return(1));
  EXPECT_CALL(mock_net, add_node(::testing::_))
    .Times(0);
  EXPECT_CALL(mock_net, update_edge(0, 1, 1.0))
    .WillOnce(::testing::Return(false));
  EXPECT_CALL(mock_net, add_edge_core(0, 1, 1.0)).Times(1);
  EXPECT_CALL(mock_net, add_edge_core(1, 0, 1.0)).Times(0);
  EXPECT_CALL(mock_net, update_edge(1, 0, 1.0)).Times(0);
  EXPECT_CALL(mock_net, degree(::testing::_))
    .Times(0);

  EXPECT_TRUE(mock_net.add_edge("1","2", 1.0));
}

TEST(TestNetwork, AddEdgeMonoGraphDirectedUpdate) {
  NetworkMockAddEdge mock_net;
  mock_net.set_multigraph(false);
  mock_net.set_directed(true);

  EXPECT_CALL(mock_net, get_idx_of_label(::testing::_))
    .WillOnce(::testing::Return(0))
    .WillOnce(::testing::Return(1));
  EXPECT_CALL(mock_net, add_node(::testing::_))
    .Times(0);
  EXPECT_CALL(mock_net, update_edge(0, 1, 1.0))
    .WillOnce(::testing::Return(true));
  EXPECT_CALL(mock_net, add_edge_core(::testing::_,::testing::_,::testing::_))
    .Times(0);
  EXPECT_CALL(mock_net, update_edge(1, 0, 1.0))
    .Times(0);
  EXPECT_CALL(mock_net, degree(::testing::_))
    .Times(0);

  EXPECT_FALSE(mock_net.add_edge("1","2", 1.0));
}

TEST(TestNetwork, AddEdgeMonoGraphUndirectedNoUpdate) {
  NetworkMockAddEdge mock_net;
  mock_net.set_multigraph(false);
  mock_net.set_directed(false);

  EXPECT_CALL(mock_net, get_idx_of_label(::testing::_))
    .WillOnce(::testing::Return(0))
    .WillOnce(::testing::Return(1));
  EXPECT_CALL(mock_net, add_node(::testing::_))
    .Times(0);
  EXPECT_CALL(mock_net, update_edge(0, 1, 1.0))
    .WillOnce(::testing::Return(false));
  EXPECT_CALL(mock_net, add_edge_core(0, 1, 1.0))
    .Times(1);
  EXPECT_CALL(mock_net, update_edge(1, 0, 1.0))
    .WillOnce(::testing::Return(false));
  EXPECT_CALL(mock_net, add_edge_core(1, 0, 1.0))
    .Times(1);
  EXPECT_CALL(mock_net, degree(0))
    .Times(1);
  EXPECT_CALL(mock_net, degree(1))
    .Times(1);

  EXPECT_TRUE(mock_net.add_edge("1","2", 1.0));
}

TEST(TestNetwork, AddEdgeMonoGraphUndirectedUpdate) {
  NetworkMockAddEdge mock_net;
  mock_net.set_multigraph(false);
  mock_net.set_directed(false);

  EXPECT_CALL(mock_net, get_idx_of_label(::testing::_))
    .WillOnce(::testing::Return(0))
    .WillOnce(::testing::Return(1));
  EXPECT_CALL(mock_net, add_node(::testing::_))
    .Times(0);
  EXPECT_CALL(mock_net, update_edge(0, 1, 1.0))
    .WillOnce(::testing::Return(true));
  EXPECT_CALL(mock_net, add_edge_core(0, 1, 1.0)).Times(0);

  EXPECT_CALL(mock_net, update_edge(1, 0, 1.0))
    .WillOnce(::testing::Return(true));
  EXPECT_CALL(mock_net, add_edge_core(1, 0, 1.0)).Times(0);
  EXPECT_CALL(mock_net, degree(0))
    .Times(1);
  EXPECT_CALL(mock_net, degree(1))
    .Times(1);

  EXPECT_FALSE(mock_net.add_edge("1","2", 1.0));
}

TEST(TestNetwork, SetDirectedFromFalseToTrue) {
  Network net;

  EXPECT_FALSE(net.is_directed());

  net.set_directed(true);
  EXPECT_TRUE(net.is_directed());
}

TEST(TestNetwork, SetDirectedFromTrueToFalse) {
  Network net;

  EXPECT_FALSE(net.is_directed());

  net.set_directed(true);
  EXPECT_TRUE(net.is_directed());

  net.set_directed(false);
  EXPECT_FALSE(net.is_directed());
}

TEST(TestNetwork, SetMultigraphFromFalseToTrue) {
  Network net;

  EXPECT_FALSE(net.is_multigraph());

  net.set_multigraph(true);
  EXPECT_TRUE(net.is_multigraph());
}

TEST(TestNetwork, SetMultigraphFromTrueToFalse) {
  Network net;

  EXPECT_FALSE(net.is_multigraph());

  net.set_multigraph(true);
  EXPECT_TRUE(net.is_multigraph());

  net.set_multigraph(false);
  EXPECT_FALSE(net.is_multigraph());
}

TEST(TestNetwork, ReadEdgeListThrowsIfEmptyFileName) {
  Network net;
  ASSERT_THROW(net.read_edge_list(""), std::runtime_error);
}

TEST(TestNetwork, ReadEdgeListThrowsIfFileCannotBeOpened) {
  Network net;
  ASSERT_THROW(net.read_edge_list("nonexistent.txt"), std::runtime_error);
}

TEST(TestNetwork, ReadEdgeListParsesUnweightedUndirectedEdgeListWithHeaders) {
  const std::string filename = "test_edges.txt";
  write_test_file(filename,
    "src\tgt\n"
    "A\tB\n"
    "A\tC\n"
    "B\tC\n"
    "C\tD\n"
  );

  Network net;
  net.read_edge_list(filename, true);  // undirected, unweighted

  // Check that all nodes label are in 'labels'
  const auto& labels = net.get_labels();
  EXPECT_EQ(labels.size(), 4);
  EXPECT_EQ(labels[0], "A");
  EXPECT_EQ(labels[1], "B");
  EXPECT_EQ(labels[2], "C");
  EXPECT_EQ(labels[3], "D");

  // Count number of edges
  EXPECT_EQ(net.get_n_edges(), 4);

  // Check symmetric edges due to undirected=true
  EXPECT_EQ(net.get_edge_weight("A", "B"), 1.0);
  EXPECT_EQ(net.get_edge_weight("A", "C"), 1.0);
  EXPECT_EQ(net.get_edge_weight("B", "A"), 1.0);
  EXPECT_EQ(net.get_edge_weight("B", "C"), 1.0);
  EXPECT_EQ(net.get_edge_weight("C", "A"), 1.0);
  EXPECT_EQ(net.get_edge_weight("C", "B"), 1.0);
  EXPECT_EQ(net.get_edge_weight("C", "D"), 1.0);
  EXPECT_EQ(net.get_edge_weight("D", "C"), 1.0);

  std::remove(filename.c_str());
}

TEST(TestNetwork, ReadEdgeListParsesUnweightedUndirectedEdgeListWithoutHeaders) {
  const std::string filename = "test_edges.txt";
  write_test_file(filename,
    "A\tB\n"
    "A\tC\n"
    "B\tC\n"
    "C\tD\n"
  );

  Network net;
  net.read_edge_list(filename, false);  // undirected, unweighted

  // Check that all nodes label are in 'labels'
  const auto& labels = net.get_labels();
  EXPECT_EQ(labels.size(), 4);
  EXPECT_EQ(labels[0], "A");
  EXPECT_EQ(labels[1], "B");
  EXPECT_EQ(labels[2], "C");
  EXPECT_EQ(labels[3], "D");

  // Count number of edges
  EXPECT_EQ(net.get_n_edges(), 4);

  // Check symmetric edges due to undirected=true
  EXPECT_EQ(net.get_edge_weight("A", "B"), 1.0);
  EXPECT_EQ(net.get_edge_weight("A", "C"), 1.0);
  EXPECT_EQ(net.get_edge_weight("B", "A"), 1.0);
  EXPECT_EQ(net.get_edge_weight("B", "C"), 1.0);
  EXPECT_EQ(net.get_edge_weight("C", "A"), 1.0);
  EXPECT_EQ(net.get_edge_weight("C", "B"), 1.0);
  EXPECT_EQ(net.get_edge_weight("C", "D"), 1.0);
  EXPECT_EQ(net.get_edge_weight("D", "C"), 1.0);

  std::remove(filename.c_str());
}

TEST(TestNetwork, ReadEdgeListParsesUnweightedUndirectedEdgeListWithoutHeadersComaSep) {
  const std::string filename = "test_edges.txt";
  write_test_file(filename,
    "A,B\n"
    "A,C\n"
    "B,C\n"
    "C,D\n"
  );

  Network net;
  net.read_edge_list(filename, false, ',');  // undirected, unweighted

  // Check that all nodes label are in 'labels'
  const auto& labels = net.get_labels();
  EXPECT_EQ(labels.size(), 4);
  EXPECT_EQ(labels[0], "A");
  EXPECT_EQ(labels[1], "B");
  EXPECT_EQ(labels[2], "C");
  EXPECT_EQ(labels[3], "D");

  // Count number of edges
  EXPECT_EQ(net.get_n_edges(), 4);

  // Check symmetric edges due to undirected=true
  EXPECT_EQ(net.get_edge_weight("A", "B"), 1.0);
  EXPECT_EQ(net.get_edge_weight("A", "C"), 1.0);
  EXPECT_EQ(net.get_edge_weight("B", "A"), 1.0);
  EXPECT_EQ(net.get_edge_weight("B", "C"), 1.0);
  EXPECT_EQ(net.get_edge_weight("C", "A"), 1.0);
  EXPECT_EQ(net.get_edge_weight("C", "B"), 1.0);
  EXPECT_EQ(net.get_edge_weight("C", "D"), 1.0);
  EXPECT_EQ(net.get_edge_weight("D", "C"), 1.0);

  std::remove(filename.c_str());
}

TEST(TestNetwork, ReadEdgeListParsesWeightedUndirectedEdgeList) {
  const std::string filename = "test_edges.txt";
  write_test_file(filename,
    "A\tB\t0.7\n"
    "A\tC\t1.2\n"
    "B\tC\t0.5\n"
    "C\tD\t1.0\n"
  );

  Network net;
  net.read_edge_list(filename);  // no headers, undirected

  // Check that all nodes label are in 'labels'
  const auto& labels = net.get_labels();
  EXPECT_EQ(labels.size(), 4);
  EXPECT_EQ(labels[0], "A");
  EXPECT_EQ(labels[1], "B");
  EXPECT_EQ(labels[2], "C");
  EXPECT_EQ(labels[3], "D");

  // Count number of edges
  EXPECT_EQ(net.get_n_edges(), 4);

  // Check symmetric edges due to undirected=true
  EXPECT_EQ(net.get_edge_weight("A", "B"), 0.7);
  EXPECT_EQ(net.get_edge_weight("A", "C"), 1.2);
  EXPECT_EQ(net.get_edge_weight("B", "A"), 0.7);
  EXPECT_EQ(net.get_edge_weight("B", "C"), 0.5);
  EXPECT_EQ(net.get_edge_weight("C", "A"), 1.2);
  EXPECT_EQ(net.get_edge_weight("C", "B"), 0.5);
  EXPECT_EQ(net.get_edge_weight("C", "D"), 1.0);
  EXPECT_EQ(net.get_edge_weight("D", "C"), 1.0);
  
  std::remove(filename.c_str());
}

TEST(TestNetwork, ReadEdgeListParsesWeightedUndirectedEdgeListWithDupEdgeHigher) {
  const std::string filename = "test_edges.txt";
  write_test_file(filename,
    "A\tB\t0.7\n"
    "A\tC\t1.2\n"
    "B\tC\t0.5\n"
    "C\tD\t1.0\n"
    "D\tC\t1.5\n"
  );

  Network net;
  net.read_edge_list(filename);  // no headers, undirected

  // Check that all nodes label are in 'labels'
  const auto& labels = net.get_labels();
  EXPECT_EQ(labels.size(), 4);
  EXPECT_EQ(labels[0], "A");
  EXPECT_EQ(labels[1], "B");
  EXPECT_EQ(labels[2], "C");
  EXPECT_EQ(labels[3], "D");

  // Count number of edges
  EXPECT_EQ(net.get_n_edges(), 4);

  // Check symmetric edges due to undirected=true
  EXPECT_EQ(net.get_edge_weight("A", "B"), 0.7);
  EXPECT_EQ(net.get_edge_weight("A", "C"), 1.2);
  EXPECT_EQ(net.get_edge_weight("B", "A"), 0.7);
  EXPECT_EQ(net.get_edge_weight("B", "C"), 0.5);
  EXPECT_EQ(net.get_edge_weight("C", "A"), 1.2);
  EXPECT_EQ(net.get_edge_weight("C", "B"), 0.5);
  EXPECT_EQ(net.get_edge_weight("C", "D"), 1.5);
  EXPECT_EQ(net.get_edge_weight("D", "C"), 1.5);

  std::remove(filename.c_str());
}

TEST(TestNetwork, ReadEdgeListParsesWeightedUndirectedEdgeListWithDupEdgeLower) {
  const std::string filename = "test_edges.txt";
  write_test_file(filename,
    "A\tB\t0.7\n"
    "A\tC\t1.2\n"
    "B\tC\t0.5\n"
    "C\tD\t1.6\n"
    "D\tC\t1.5\n"
  );

  Network net;
  net.read_edge_list(filename);  // no headers, undirected

  // Check that all nodes label are in 'labels'
  const auto& labels = net.get_labels();
  EXPECT_EQ(labels.size(), 4);
  EXPECT_EQ(labels[0], "A");
  EXPECT_EQ(labels[1], "B");
  EXPECT_EQ(labels[2], "C");
  EXPECT_EQ(labels[3], "D");

  // Count number of edges
  EXPECT_EQ(net.get_n_edges(), 4);

  // Check symmetric edges due to undirected=true
  EXPECT_EQ(net.get_edge_weight("A", "B"), 0.7);
  EXPECT_EQ(net.get_edge_weight("A", "C"), 1.2);
  EXPECT_EQ(net.get_edge_weight("B", "A"), 0.7);
  EXPECT_EQ(net.get_edge_weight("B", "C"), 0.5);
  EXPECT_EQ(net.get_edge_weight("C", "A"), 1.2);
  EXPECT_EQ(net.get_edge_weight("C", "B"), 0.5);
  EXPECT_EQ(net.get_edge_weight("C", "D"), 1.6);
  EXPECT_EQ(net.get_edge_weight("D", "C"), 1.6);

  std::remove(filename.c_str());
}

TEST(TestNetwork, ReadEdgeListParsesUnweightedDirectedEdgeList) {
  const std::string filename = "test_edges.txt";
  write_test_file(filename,
    "A\tC\n"
    "A\tD\n"
    "B\tA\n"
    "B\tC\n"
    "C\tD\n"
    "D\tC\n"
  );

  Network net;
  net.read_edge_list(filename, false, '\t', true);  // no heades, directed

  // Check that all nodes label are in 'labels'
  const auto& labels = net.get_labels();
  EXPECT_EQ(labels.size(), 4);
  EXPECT_EQ(labels[0], "A");
  EXPECT_EQ(labels[1], "B");
  EXPECT_EQ(labels[2], "C");
  EXPECT_EQ(labels[3], "D");

  // Count number of edges
  EXPECT_EQ(net.get_n_edges(), 6);

  // Check symmetric edges due to undirected=true
  EXPECT_EQ(net.get_edge_weight("A", "C"), 1.0);
  EXPECT_EQ(net.get_edge_weight("A", "D"), 1.0);
  EXPECT_EQ(net.get_edge_weight("B", "A"), 1.0);
  EXPECT_EQ(net.get_edge_weight("B", "C"), 1.0);
  EXPECT_EQ(net.get_edge_weight("C", "D"), 1.0);
  EXPECT_EQ(net.get_edge_weight("D", "C"), 1.0);

  std::remove(filename.c_str());
}

TEST(TestNetwork, ReadEdgeListParsesWeightedDirectedEdgeList) {
  const std::string filename = "test_edges.txt";
  write_test_file(filename,
      "A\tB\t0.9\n"
      "A\tC\t0.1\n"
      "B\tA\t0.3\n"
      "B\tC\t0.5\n"
      "B\tD\t0.2\n"
      "D\tC\t1.0\n"
      "E\tA\t1.0\n"
  );

  Network net;
  net.read_edge_list(filename, false, '\t', true);  // undirected, weighted

  // Check that all nodes label are in 'labels'
  const auto& labels = net.get_labels();
  EXPECT_EQ(labels.size(), 5);
  EXPECT_EQ(labels[0], "A");
  EXPECT_EQ(labels[1], "B");
  EXPECT_EQ(labels[2], "C");
  EXPECT_EQ(labels[3], "D");
  EXPECT_EQ(labels[4], "E");

  // Count number of edges
  EXPECT_EQ(net.get_n_edges(), 7);

  // Check symmetric edges due to undirected=true
  EXPECT_EQ(net.get_edge_weight("A", "B"), 0.9);
  EXPECT_EQ(net.get_edge_weight("A", "C"), 0.1);
  EXPECT_EQ(net.get_edge_weight("B", "A"), 0.3);
  EXPECT_EQ(net.get_edge_weight("B", "C"), 0.5);
  EXPECT_EQ(net.get_edge_weight("B", "D"), 0.2);
  EXPECT_EQ(net.get_edge_weight("D", "C"), 1.0);
  EXPECT_EQ(net.get_edge_weight("E", "A"), 1.0);

  std::remove(filename.c_str());
}

TEST(TestNetwork, TestNetwork_GetColSumsCreatesEmptyVectorWhenN_ColsIszZer_Test) {
  Network net;
  std::vector<double> col_sums;
  net.get_col_sums(col_sums);

  EXPECT_TRUE(col_sums.empty());
}

TEST(TestNetwork, GetColSumsCorrectlyCalculateSumsDefaultLabel) {
  Network net;
  net.add_nodes({"A","B","C","D"});
  net.add_edge("A","B",1.1);
  net.add_edge("A","C",0.7);
  net.add_edge("B","D",0.6);
  net.add_edge("C","D",0.5);

  std::vector<double> col_sums;
  net.get_col_sums(col_sums);

  ASSERT_EQ(col_sums.size(), 4);
  EXPECT_NEAR(col_sums[0], 1.8, 1e-8);
  EXPECT_NEAR(col_sums[1], 1.7, 1e-8);
  EXPECT_NEAR(col_sums[2], 1.2, 1e-8);
  EXPECT_NEAR(col_sums[3], 1.1, 1e-8);
}

TEST(TestNetwork, GetColSumsCorrectlyCalculateSumsNodefaultLabel) {
  Network net;
  net.add_nodes({"A","B","C","D"});
  net.add_edge("A","B",1.1);
  net.add_edge("A","C",0.7);
  net.add_edge("B","D",0.6);
  net.add_edge("C","D",0.5);

  std::vector<double> col_sums;
  std::vector<std::string> label_list = {"A","B","C","E"};
  net.get_col_sums(col_sums, label_list);

  ASSERT_EQ(col_sums.size(), 4);
  EXPECT_EQ(col_sums[0], 1.8);
  EXPECT_EQ(col_sums[1], 1.1);
  EXPECT_EQ(col_sums[2], 0.7);
  EXPECT_EQ(col_sums[3], 0.0);
}

TEST(TestNetwork, GetEdgeWeightReturnsNanWhenSrcDoesNotExist) {
  Network net;
  net.add_nodes({"A","B","C"});
  net.add_edge("A","B");
  net.add_edge("A","C");

  EXPECT_TRUE(std::isnan(net.get_edge_weight("D","B")));
}

TEST(TestNetwork, GetEdgeWeightReturnsNanWhenTgtDoesNotExist) {
  Network net;
  net.add_nodes({"A","B","C"});
  net.add_edge("A","B");
  net.add_edge("A","C");

  EXPECT_TRUE(std::isnan(net.get_edge_weight("A","E")));
}

TEST(TestNetwork, GetEdgeWeightReturnsNanWhenEdgeDoesNotExist) {
  Network net;
  net.add_nodes({"A","B","C"});
  net.add_edge("A","B");
  net.add_edge("A","C");

  EXPECT_TRUE(std::isnan(net.get_edge_weight("B","C")));
}

TEST(TestNetwork, GetEdgeWeightReturnCorrectWeight) {
  Network net;
  net.add_nodes({"A","B","C"});
  net.add_edge("A","B", 1.2);
  net.add_edge("A","C");

  EXPECT_EQ(net.get_edge_weight("A","B"), 1.2);
}

TEST(TestNetwork, GetAdjacencyMatrixDefaultInput) {
  Network net;
  net.add_nodes({"A","B","C","D"});
  net.add_edge("A", "B", 0.7);
  net.add_edge("A", "C", 1.2);
  net.add_edge("B", "C", 0.5);
  net.add_edge("C", "D", 1.0);

  CSR_Matrix mat = net.get_adjacency_matrix();

  EXPECT_EQ(mat.n_rows(), 4);
  EXPECT_EQ(mat.n_cols(), 4);
  EXPECT_EQ(mat.nnz(), 8);

  auto actual_values = mat.get_values();
  auto actual_col_idx = mat.get_col_idx();
  auto actual_row_ptr = mat.get_row_ptr();

  ASSERT_EQ(actual_values.size(), 8);
  EXPECT_EQ(actual_values[0], 0.7);
  EXPECT_EQ(actual_values[1], 1.2);
  EXPECT_EQ(actual_values[2], 0.7);
  EXPECT_EQ(actual_values[3], 0.5);
  EXPECT_EQ(actual_values[4], 1.2);
  EXPECT_EQ(actual_values[5], 0.5);
  EXPECT_EQ(actual_values[6], 1.0);
  EXPECT_EQ(actual_values[7], 1.0);

  ASSERT_EQ(actual_col_idx.size(), 8);
  EXPECT_EQ(actual_col_idx[0], 1);
  EXPECT_EQ(actual_col_idx[1], 2);
  EXPECT_EQ(actual_col_idx[2], 0);
  EXPECT_EQ(actual_col_idx[3], 2);
  EXPECT_EQ(actual_col_idx[4], 0);
  EXPECT_EQ(actual_col_idx[5], 1);
  EXPECT_EQ(actual_col_idx[6], 3);
  EXPECT_EQ(actual_col_idx[7], 2);

  ASSERT_EQ(actual_row_ptr.size(), 5);
  EXPECT_EQ(actual_row_ptr[0], 0);
  EXPECT_EQ(actual_row_ptr[1], 2);
  EXPECT_EQ(actual_row_ptr[2], 4);
  EXPECT_EQ(actual_row_ptr[3], 7);
  EXPECT_EQ(actual_row_ptr[4], 8);
}

TEST(TestNetwork, GetAdjacencyMatrixSubsetOfNodes) {
  Network net;
  net.add_nodes({"A","B","C","D"});
  net.add_edge("A", "B", 0.7);
  net.add_edge("A", "C", 1.2);
  net.add_edge("B", "C", 0.5);
  net.add_edge("C", "D", 1.0);

  std::vector<std::string> label_list = {"A", "C", "D"};
  CSR_Matrix mat = net.get_adjacency_matrix(label_list);

  EXPECT_EQ(mat.n_rows(), 3);
  EXPECT_EQ(mat.n_cols(), 3);
  EXPECT_EQ(mat.nnz(), 4);

  auto actual_values = mat.get_values();
  auto actual_col_idx = mat.get_col_idx();
  auto actual_row_ptr = mat.get_row_ptr();

  ASSERT_EQ(actual_values.size(), 4);
  EXPECT_EQ(actual_values[0], 1.2);
  EXPECT_EQ(actual_values[1], 1.2);
  EXPECT_EQ(actual_values[2], 1.0);
  EXPECT_EQ(actual_values[3], 1.0);

  ASSERT_EQ(actual_col_idx.size(), 4);
  EXPECT_EQ(actual_col_idx[0], 1);
  EXPECT_EQ(actual_col_idx[1], 0);
  EXPECT_EQ(actual_col_idx[2], 2);
  EXPECT_EQ(actual_col_idx[3], 1);

  ASSERT_EQ(actual_row_ptr.size(), 4);
  EXPECT_EQ(actual_row_ptr[0], 0);
  EXPECT_EQ(actual_row_ptr[1], 1);
  EXPECT_EQ(actual_row_ptr[2], 3);
  EXPECT_EQ(actual_row_ptr[3], 4);
}

TEST(TestNetwork, GetAdjacencyMatrixSupersetOfNodes) {
  Network net;
  net.add_nodes({"A","B","C","D"});
  net.add_edge("A", "B", 0.7);
  net.add_edge("A", "C", 1.2);
  net.add_edge("B", "C", 0.5);
  net.add_edge("C", "D", 1.0);

  std::vector<std::string> label_list = {"A", "B", "C", "D", "E"};
  CSR_Matrix mat = net.get_adjacency_matrix(label_list);

  EXPECT_EQ(mat.n_rows(), 5);
  EXPECT_EQ(mat.n_cols(), 5);
  EXPECT_EQ(mat.nnz(), 8);

  auto actual_values = mat.get_values();
  auto actual_col_idx = mat.get_col_idx();
  auto actual_row_ptr = mat.get_row_ptr();

  ASSERT_EQ(actual_values.size(), 8);
  EXPECT_EQ(actual_values[0], 0.7);
  EXPECT_EQ(actual_values[1], 1.2);
  EXPECT_EQ(actual_values[2], 0.7);
  EXPECT_EQ(actual_values[3], 0.5);
  EXPECT_EQ(actual_values[4], 1.2);
  EXPECT_EQ(actual_values[5], 0.5);
  EXPECT_EQ(actual_values[6], 1.0);
  EXPECT_EQ(actual_values[7], 1.0);

  ASSERT_EQ(actual_col_idx.size(), 8);
  EXPECT_EQ(actual_col_idx[0], 1);
  EXPECT_EQ(actual_col_idx[1], 2);
  EXPECT_EQ(actual_col_idx[2], 0);
  EXPECT_EQ(actual_col_idx[3], 2);
  EXPECT_EQ(actual_col_idx[4], 0);
  EXPECT_EQ(actual_col_idx[5], 1);
  EXPECT_EQ(actual_col_idx[6], 3);
  EXPECT_EQ(actual_col_idx[7], 2);

  ASSERT_EQ(actual_row_ptr.size(), 6);
  EXPECT_EQ(actual_row_ptr[0], 0);
  EXPECT_EQ(actual_row_ptr[1], 2);
  EXPECT_EQ(actual_row_ptr[2], 4);
  EXPECT_EQ(actual_row_ptr[3], 7);
  EXPECT_EQ(actual_row_ptr[4], 8);
  EXPECT_EQ(actual_row_ptr[5], 8);
}

TEST(TestNetwork, GetAdjacencyMatrixMixutureOfPresentAndMissingNodes) {
 Network net;
  net.add_nodes({"A","D","C","B"});
  net.add_edge("A", "B", 0.7);
  net.add_edge("A", "C", 1.2);
  net.add_edge("B", "C", 0.5);
  net.add_edge("C", "D", 1.0);

  std::vector<std::string> label_list = {"A", "B", "C", "E"};
  CSR_Matrix mat = net.get_adjacency_matrix(label_list);

  EXPECT_EQ(mat.n_rows(), 4);
  EXPECT_EQ(mat.n_cols(), 4);
  EXPECT_EQ(mat.nnz(), 6);

  auto actual_values = mat.get_values();
  auto actual_col_idx = mat.get_col_idx();
  auto actual_row_ptr = mat.get_row_ptr();

  ASSERT_EQ(actual_values.size(), 6);
  EXPECT_EQ(actual_values[0], 0.7);
  EXPECT_EQ(actual_values[1], 1.2);
  EXPECT_EQ(actual_values[2], 0.7);
  EXPECT_EQ(actual_values[3], 0.5);
  EXPECT_EQ(actual_values[4], 1.2);
  EXPECT_EQ(actual_values[5], 0.5);

  ASSERT_EQ(actual_col_idx.size(), 6);
  EXPECT_EQ(actual_col_idx[0], 1);
  EXPECT_EQ(actual_col_idx[1], 2);
  EXPECT_EQ(actual_col_idx[2], 0);
  EXPECT_EQ(actual_col_idx[3], 2);
  EXPECT_EQ(actual_col_idx[4], 0);
  EXPECT_EQ(actual_col_idx[5], 1);

  ASSERT_EQ(actual_row_ptr.size(), 5);
  EXPECT_EQ(actual_row_ptr[0], 0);
  EXPECT_EQ(actual_row_ptr[1], 2);
  EXPECT_EQ(actual_row_ptr[2], 4);
  EXPECT_EQ(actual_row_ptr[3], 6);
  EXPECT_EQ(actual_row_ptr[4], 6);
}

TEST(TestNetwork, GetTransitionMatrixThrowsOnDirected) {
  Network net;
  net.set_directed(true);

  CSR_Matrix mat;
  ASSERT_THAT(
    [&](){net.get_transition_matrix(); },
    testing::ThrowsMessage<std::runtime_error>("Network::get_transition_matrix - transition matrix is not implemented for directed networks")
  );
}

TEST(TestNetwork, GetTransitionMatrixCallsAdjacencyAndNormalizes) {
  NetworkMockGetAdjMatrix mock_net;
  std::vector<std::string> labels = {"A", "B"};

  // Create a mock CSR_Matrix
  CSR_Matrix mock_matrix = CSR_Matrix::diag(2, 1.5);

  
  EXPECT_CALL(mock_net, get_adjacency_matrix(labels))
    .WillOnce(::testing::Return(mock_matrix));

  CSR_Matrix result = mock_net.get_transition_matrix(labels);

  EXPECT_EQ(result.n_rows(), 2);
  EXPECT_EQ(result.n_cols(), 2);
  EXPECT_EQ(result.nnz(), 2);

  auto actual_values = result.get_values();
  auto actual_col_idx = result.get_col_idx();
  auto actual_row_ptr = result.get_row_ptr();

  ASSERT_EQ(actual_values.size(), 2);
  EXPECT_EQ(actual_values[0], 1.0);
  EXPECT_EQ(actual_values[1], 1.0);

  ASSERT_EQ(actual_col_idx.size(), 2);
  EXPECT_EQ(actual_col_idx[0], 0);
  EXPECT_EQ(actual_col_idx[1], 1);

  ASSERT_EQ(actual_row_ptr.size(), 3);
  EXPECT_EQ(actual_row_ptr[0], 0);
  EXPECT_EQ(actual_row_ptr[1], 1);
  EXPECT_EQ(actual_row_ptr[2], 2);
}

TEST(TestNetwork, PrintThrowsIfFileCouldNotBeOpened) {
  Network net;

  std::string bad_filename = "/invalid/path/to/file.txt";

  ASSERT_THAT(
    [&](){net.print(bad_filename); },
    testing::ThrowsMessage<std::runtime_error>("Network::print - Could not open file: /invalid/path/to/file.txt")
  );
}

TEST(TestNetwork, PrintRecordsDataCorrectly) {
  Network net;
  net.add_nodes({"A","C","B","D"});
  net.add_edge("A", "B", 0.7);
  net.add_edge("A", "C", 1.2);
  net.add_edge("B", "C", 0.5);
  net.add_edge("C", "D", 1.0);

  std::string file_name = "test_networktxt";
  net.print(file_name);

  std::ifstream file(file_name);
  ASSERT_TRUE(file.is_open()) << "Could not open output file";

  std::string line;
  std::vector<std::string> lines;
  while (std::getline(file, line)) {
    lines.push_back(line);
  }

  file.close();
  std::remove(file_name.c_str());

  ASSERT_EQ(lines.size(), 17);
  EXPECT_EQ(lines[0], "labels_ (label → index):");

  // Search within label section for expected label lines
  std::vector<std::string> expected_labels = {
    "A: 0",
    "B: 2",
    "C: 1",
    "D: 3"
  };

  // Check that label -> index pairs are present starting 1 line after section start
  // and only looking 4 lines down
  for (const auto& expected : expected_labels) {
    auto found = std::find(lines.begin() + 2, lines.end() + 6, expected);
    EXPECT_NE(found, lines.end()) << "Missing label line: " << expected;
  }
  
  // Check index_to_label section
  EXPECT_EQ(lines[5], "");
  EXPECT_EQ(lines[6], "index_to_label_ (index → label):");
  EXPECT_EQ(lines[7], "0: A");
  EXPECT_EQ(lines[8], "1: C");
  EXPECT_EQ(lines[9], "2: B");
  EXPECT_EQ(lines[10], "3: D");

  // Check edges section
  EXPECT_EQ(lines[11], "");
  EXPECT_EQ(lines[12], "edges_ (src_index → [target_index, weight]):");
  EXPECT_EQ(lines[13], "0: [2, 0.7] [1, 1.2] ");
  EXPECT_EQ(lines[14], "1: [0, 1.2] [2, 0.5] [3, 1] ");
  EXPECT_EQ(lines[15], "2: [0, 0.7] [1, 0.5] ");
  EXPECT_EQ(lines[16], "3: [1, 1] ");
}

TEST(TestNetwork, AddEdgeCoreThrowsOnOutOfRange) {
  NetworkProtected net;
  net.add_nodes({"1", "2", "3"});

  ASSERT_THAT(
    [&](){net.add_edge_core(5, 1, 1.0); },
    testing::ThrowsMessage<std::out_of_range>("Network::add_edge_core - src_idx is out of range")
  );
}

TEST(TestNetwork, AddEdgeCoreAddsEdge) {
  NetworkProtected net;
  net.add_nodes({"1", "2", "3"});

  net.add_edge_core(0, 1, 1.0);
  const std::size_t actual_nnz = net.get_nnz();
  EXPECT_EQ(actual_nnz, 1);
}

TEST(TestNetwork, UpdateEdgeThrowsOnSrcOutOfRange) {
  NetworkProtected net;
  net.add_nodes({"1", "2", "3"});
  net.add_edge("1", "2");

  ASSERT_THAT(
    [&](){net.update_edge(5, 1, 1.0); },
    testing::ThrowsMessage<std::out_of_range>("Network::update_edge - src_idx is out of range")
  );
}

TEST(TestNetwork, UpdateEdgeThrowsOnTgtOutOfRange) {
  NetworkProtected net;
  net.add_nodes({"1", "2", "3"});
  net.add_edge("1", "2");

  ASSERT_THAT(
    [&](){net.update_edge(0, 5, 1.0); },
    testing::ThrowsMessage<std::out_of_range>("Network::update_edge - tgt_idx is out of range")
  );
}

TEST(TestNetwork, UpdateEdgeUpdatesEdge) {
  NetworkProtected net;
  net.add_nodes({"1", "2", "3"});
  net.add_edge_core(0, 1, 1.0);
  net.add_edge_core(1, 0, 1.0);

  bool updated = net.update_edge(0, 1, 3.0);

  EXPECT_EQ(updated, true);

  const std::size_t acutal_num_edges = net.get_n_edges();
  EXPECT_EQ(acutal_num_edges, 1);

  const std::size_t actual_nnz = net.get_nnz();
  EXPECT_EQ(actual_nnz, 2);

  EXPECT_NEAR(net.get_edge_weight("1", "2"), 3.0, 1e-8);
  EXPECT_NEAR(net.get_edge_weight("2", "1"), 1.0, 1e-8);
}

TEST(TestNetwork, UpdateEdgeDoesNotUpdatesEdgeWithHigherWeight) {
  NetworkProtected net;
  net.add_nodes({"1", "2", "3"});
  net.add_edge_core(0, 1, 1.0);
  net.add_edge_core(1, 0, 1.0);

  bool updated = net.update_edge(0, 1, 3.0);

  EXPECT_EQ(updated, true);

  const std::size_t acutal_num_edges = net.get_n_edges();
  EXPECT_EQ(acutal_num_edges, 1);

  const std::size_t actual_nnz = net.get_nnz();
  EXPECT_EQ(actual_nnz, 2);

  EXPECT_NEAR(net.get_edge_weight("1", "2"), 3.0, 1e-8);
  EXPECT_NEAR(net.get_edge_weight("2", "1"), 1.0, 1e-8);
}

TEST(TestNetwork, UpdateEdgeReturnsFalseIfEdgeNotFound) {
  NetworkProtected net;
  net.add_nodes({"1", "2", "3"});
  net.add_edge_core(0, 1, 1.0);
  net.add_edge_core(1, 0, 1.0);

  bool updated = net.update_edge(0, 2, 3.0);

  EXPECT_EQ(updated, false);

  const std::size_t acutal_num_edges = net.get_n_edges();
  EXPECT_EQ(acutal_num_edges, 1);

  const std::size_t actual_nnz = net.get_nnz();
  EXPECT_EQ(actual_nnz, 2);

  EXPECT_NEAR(net.get_edge_weight("1", "2"), 1.0, 1e-8);
  EXPECT_NEAR(net.get_edge_weight("2", "1"), 1.0, 1e-8);
}

TEST(TestNetwork, CreateLocalLabelListCallsGetLabelsWhenInputIsEmpty) {
  NetworkProtected net;
  net.add_nodes({"A", "B", "C"});
  
  std::vector<std::string> label_list;
  const auto actual_local_labels = net.create_local_label_list(label_list);

  ASSERT_EQ(actual_local_labels.size(), 3);
  EXPECT_EQ(actual_local_labels[0], "A");
  EXPECT_EQ(actual_local_labels[1], "B");
  EXPECT_EQ(actual_local_labels[2], "C");
}

TEST(TestNetwork, CreateLocalLabelListReturnsInputWhenNonEmpty) {
  NetworkProtected net;
  net.add_nodes({"A", "B", "C"});
  
  std::vector<std::string> label_list = {"A","B","C","D"};
  const auto actual_local_labels = net.create_local_label_list(label_list);

  ASSERT_EQ(actual_local_labels.size(), 4);
  EXPECT_EQ(actual_local_labels[0], "A");
  EXPECT_EQ(actual_local_labels[1], "B");
  EXPECT_EQ(actual_local_labels[2], "C");
  EXPECT_EQ(actual_local_labels[3], "D");
}

TEST(TestNetwork, GetTransitionMatrixSizeReturnsCorrectSizeOnEmptyLabelList) {
  NetworkProtected net;
  net.add_nodes({"A","B","C","D"});
  net.add_edge("A","B",0.8);
  net.add_edge("A","C",0.9);
  net.add_edge("B","C",0.5);
  net.add_edge("B","D",1.0);
  net.add_edge("C","D",1.0);

  std::size_t n_rows, n_cols, nnz;
  std::vector<std::string> label_list;
  net.get_transition_matrix_size(n_rows, n_cols, nnz, label_list);

  EXPECT_EQ(n_rows, 4);
  EXPECT_EQ(n_cols, 4);
  EXPECT_EQ(nnz, 10);
}

TEST(TestNetwork, GetTransitionMatrixSizeReturnsCorrectSizeOnNonEmptyLabelList) {
  NetworkProtected net;
  net.add_nodes({"A","B","C","D"});
  net.add_edge("A","B",0.8);
  net.add_edge("A","C",0.9);
  net.add_edge("B","C",0.5);
  net.add_edge("B","D",1.0);
  net.add_edge("C","D",1.0);

  std::size_t n_rows, n_cols, nnz;
  std::vector<std::string> label_list = {"A","B","C","E","F"};
  net.get_transition_matrix_size(n_rows, n_cols, nnz, label_list);

  EXPECT_EQ(n_rows, 5);
  EXPECT_EQ(n_cols, 5);
  EXPECT_EQ(nnz, 6);
}

TEST(TestNetwork, MergeMethodFromStringReturnsEnumValueOnValidString) {
  EXPECT_EQ(MergeMethod::Max, NetworkProtected::merge_method_from_string("max"));
  EXPECT_EQ(MergeMethod::Min, NetworkProtected::merge_method_from_string("min"));
  EXPECT_EQ(MergeMethod::All, NetworkProtected::merge_method_from_string("all"));
  EXPECT_EQ(MergeMethod::Sum, NetworkProtected::merge_method_from_string("sum"));
  EXPECT_EQ(MergeMethod::Mean, NetworkProtected::merge_method_from_string("mean"));
}

TEST(TestNetwork, MergeMethodFromStringThrowsOnUnkownString) {
  ASSERT_THAT(
    [&](){NetworkProtected::merge_method_from_string("unk"); },
    testing::ThrowsMessage<std::invalid_argument>("Network::merge_method_from_string - Invalid merge method: unk. Expected: max, min, all, sum, mean.")
  );
}

TEST(TestNetwork, MergeMethodToStringReturnsExpectedString) {
  EXPECT_EQ("max", NetworkProtected::merge_method_to_string(MergeMethod::Max));
  EXPECT_EQ("min", NetworkProtected::merge_method_to_string(MergeMethod::Min));
  EXPECT_EQ("all", NetworkProtected::merge_method_to_string(MergeMethod::All));
  EXPECT_EQ("sum", NetworkProtected::merge_method_to_string(MergeMethod::Sum));
  EXPECT_EQ("mean", NetworkProtected::merge_method_to_string(MergeMethod::Mean));
}

TEST(TestNetwork, PackUnpackEdge) {
  uint32_t src = 12345, tgt = 67890;
  uint64_t key = NetworkProtected::pack_edge(src, tgt);
  uint32_t src2 = key >> 32;
  uint32_t tgt2 = key & 0xffffffffu;
  EXPECT_EQ(src, src2);
  EXPECT_EQ(tgt, tgt2);
}

class MergeLayersTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Network 1
    n1.add_nodes({"A","B","C"});
    n1.add_edge("A", "B", 1.0);
    n1.add_edge("A", "C", 2.0);
    n1.add_edge("B", "C", 3.0);

    // Network 2
    n2.add_nodes({"B","C","D"});
    n2.add_edge("B", "C", 4.0);
    n2.add_edge("C", "D", 5.0);

    nets = {n1, n2};
  }

    Network n1, n2;
    std::vector<Network> nets;
};

TEST_F(MergeLayersTest, MaxMerge) {
  Network merged = Network::merge_networks(nets, MergeMethod::Max);

  std::vector<std::string> expected_labels = {"A","B","C","D"};
  auto actual_labels = merged.get_labels();
  EXPECT_EQ(actual_labels, expected_labels);

  EXPECT_EQ(4, merged.get_n_edges());

  EXPECT_EQ(merged.get_edge_weight("A", "B"), 1.0);
  EXPECT_EQ(merged.get_edge_weight("B", "A"), 1.0);
  EXPECT_EQ(merged.get_edge_weight("A", "C"), 2.0);
  EXPECT_EQ(merged.get_edge_weight("C", "A"), 2.0);
  EXPECT_EQ(merged.get_edge_weight("B", "C"), 4.0);
  EXPECT_EQ(merged.get_edge_weight("C", "B"), 4.0);
  EXPECT_EQ(merged.get_edge_weight("C", "D"), 5.0);
  EXPECT_EQ(merged.get_edge_weight("D", "C"), 5.0);
}

TEST_F(MergeLayersTest, MinMerge) {
  Network merged = Network::merge_networks(nets, MergeMethod::Min);

  std::vector<std::string> expected_labels = {"A","B","C","D"};
  auto actual_labels = merged.get_labels();
  EXPECT_EQ(actual_labels, expected_labels);

  EXPECT_EQ(4, merged.get_n_edges());

  EXPECT_EQ(merged.get_edge_weight("A", "B"), 1.0);
  EXPECT_EQ(merged.get_edge_weight("B", "A"), 1.0);
  EXPECT_EQ(merged.get_edge_weight("A", "C"), 2.0);
  EXPECT_EQ(merged.get_edge_weight("C", "A"), 2.0);
  EXPECT_EQ(merged.get_edge_weight("B", "C"), 3.0);
  EXPECT_EQ(merged.get_edge_weight("C", "B"), 3.0);
  EXPECT_EQ(merged.get_edge_weight("C", "D"), 5.0);
  EXPECT_EQ(merged.get_edge_weight("D", "C"), 5.0);
}

TEST_F(MergeLayersTest, SumMerge) {
  Network merged = Network::merge_networks(nets, MergeMethod::Sum);

  std::vector<std::string> expected_labels = {"A","B","C","D"};
  auto actual_labels = merged.get_labels();
  EXPECT_EQ(actual_labels, expected_labels);

  EXPECT_EQ(4, merged.get_n_edges());

  EXPECT_EQ(merged.get_edge_weight("A", "B"), 1.0);
  EXPECT_EQ(merged.get_edge_weight("B", "A"), 1.0);
  EXPECT_EQ(merged.get_edge_weight("A", "C"), 2.0);
  EXPECT_EQ(merged.get_edge_weight("C", "A"), 2.0);
  EXPECT_EQ(merged.get_edge_weight("B", "C"), 7.0);
  EXPECT_EQ(merged.get_edge_weight("C", "B"), 7.0);
  EXPECT_EQ(merged.get_edge_weight("C", "D"), 5.0);
  EXPECT_EQ(merged.get_edge_weight("D", "C"), 5.0);
}

TEST_F(MergeLayersTest, MeanMerge) {
  Network merged = Network::merge_networks(nets, MergeMethod::Mean);

  std::vector<std::string> expected_labels = {"A","B","C","D"};
  auto actual_labels = merged.get_labels();
  EXPECT_EQ(actual_labels, expected_labels);

  EXPECT_EQ(4, merged.get_n_edges());

  EXPECT_EQ(merged.get_edge_weight("A", "B"), 1.0);
  EXPECT_EQ(merged.get_edge_weight("B", "A"), 1.0);
  EXPECT_EQ(merged.get_edge_weight("A", "C"), 2.0);
  EXPECT_EQ(merged.get_edge_weight("C", "A"), 2.0);
  EXPECT_EQ(merged.get_edge_weight("B", "C"), 3.5);
  EXPECT_EQ(merged.get_edge_weight("C", "B"), 3.5);
  EXPECT_EQ(merged.get_edge_weight("C", "D"), 5.0);
  EXPECT_EQ(merged.get_edge_weight("D", "C"), 5.0);
}

TEST(TestNetwork, ConvertsEdgeWeightIfWeightsLessThanOrEqualToOne) {
  Network n;
  n.add_nodes({"A","B","C"});
  n.add_edge("A", "B", 1.0);
  n.add_edge("A", "C", 0.7);
  n.add_edge("B", "C", 0.01);

  n.convert_edges_to_distance();

  EXPECT_DOUBLE_EQ(n.get_edge_weight("A", "B"), 0.0);
  EXPECT_DOUBLE_EQ(n.get_edge_weight("B", "A"), 0.0);
  EXPECT_DOUBLE_EQ(n.get_edge_weight("A", "C"), 0.3);
  EXPECT_DOUBLE_EQ(n.get_edge_weight("C", "A"), 0.3);
  EXPECT_DOUBLE_EQ(n.get_edge_weight("B", "C"), 0.99);
  EXPECT_DOUBLE_EQ(n.get_edge_weight("C", "B"), 0.99);
}

TEST(TestNetwork, ConvertEdgesToDistanceThrowsIfWeightGreatherThanOne) {
  Network n;
  n.add_nodes({"A","B","C"});
  n.add_edge("A", "B", 1.0);
  n.add_edge("A", "C", 0.7);
  n.add_edge("B", "C", 1.01);

  ASSERT_THAT(
    [&](){n.convert_edges_to_distance(); },
    testing::ThrowsMessage<std::runtime_error>("Network::convert_edges_to_distance - Cannot calculate distance for edge weight > 1.0")
  );
}

TEST(TestNetwork, ConvertPathsToLabelsEmptyInputReturnsEmptyOutput) {
  NetworkProtected net;
  net.add_nodes({"A", "B", "C", "D"});

  std::vector<std::vector<uint32_t>> input;
  auto result = net.convert_paths_to_labels(input);

  EXPECT_TRUE(result.empty());
}

TEST(TestNetwork, ConvertPathsToLabelsSingleNodePath) {
  NetworkProtected net;
  net.add_nodes({"A", "B", "C", "D"});

  std::vector<std::vector<uint32_t>> input = {
    {0}
  };
  auto result = net.convert_paths_to_labels(input);

  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0].size(), 1);
  EXPECT_EQ(result[0][0], "A");
}

TEST(TestNetwork, ConvertPathsToLabelsMultiplePathsDifferentLengths) {
  NetworkProtected net;
  net.add_nodes({"A", "B", "C", "D"});

  std::vector<std::vector<uint32_t>> input = {
    {0, 1, 2},
    {2, 3},
    {1}
  };

  auto result = net.convert_paths_to_labels(input);

  ASSERT_EQ(result.size(), 3);

  EXPECT_EQ(result[0], std::vector<std::string>({"A", "B", "C"}));
  EXPECT_EQ(result[1], std::vector<std::string>({"C", "D"}));
  EXPECT_EQ(result[2], std::vector<std::string>({"B"}));
}

TEST(TestNetwork, ConvertPathsToLabelsPreservesOrder) {
  NetworkProtected net;
  net.add_nodes({"A", "B", "C", "D"});

  std::vector<std::vector<uint32_t>> input = {
    {3,1,2,0}
  };
  auto result = net.convert_paths_to_labels(input);

  ASSERT_EQ(result.size(), 1);
  EXPECT_EQ(result[0], std::vector<std::string>({"D", "B", "C", "A"}));
}

TEST(TestNetwork, ReconstructPathsSinglePath) {
  NetworkProtected net;

  uint32_t source = 0 ;
  std::unordered_set<uint32_t> targets = {2};
  std::vector<std::vector<uint32_t>> preds(3);
  preds[1] = {0};
  preds[2] = {1};

  std::vector<std::vector<uint32_t>> result;
  net.reconstruct_paths(source, targets, preds, result);

  ASSERT_EQ(result.size(), 1);
  EXPECT_EQ(result[0], (std::vector<uint32_t>{0, 1, 2}));
}

TEST(TestNetwork, ReconstructPathsMultiplePaths) {
  NetworkProtected net;

  uint32_t source = 0;
  std::unordered_set<uint32_t> targets = {3};

  std::vector<std::vector<uint32_t>> preds(4);
  preds[1] = {0};
  preds[2] = {0};
  preds[3] = {1, 2};

  std::vector<std::vector<uint32_t>> result;

  net.reconstruct_paths(source, targets, preds, result);

  ASSERT_EQ(result.size(), 2);

  // Because order is undefined, check set membership.
  std::vector<std::vector<uint32_t>> expected = {
    {0, 1, 3},
    {0, 2, 3}
  };

  // Convert result for comparison
  auto sort_paths = [](auto& paths) {
    std::sort(paths.begin(), paths.end());
  };

  sort_paths(result);
  sort_paths(expected);

  EXPECT_EQ(result, expected);
}

TEST(TestNetwork, ReconstructPathsUnreachableTarget) {
  NetworkProtected net;

  uint32_t source = 0;
  std::unordered_set<uint32_t> targets = {5};  // No preds

  std::vector<std::vector<uint32_t>> preds;

  std::vector<std::vector<uint32_t>> result;

  net.reconstruct_paths(source, targets, preds, result);

  EXPECT_TRUE(result.empty());
}

TEST(TestNetwork, ReconstructPathsMultipleTargets) {
  NetworkProtected net;

  uint32_t source = 0;
  std::unordered_set<uint32_t> targets = {2, 4};

  std::vector<std::vector<uint32_t>> preds(5);
  preds[1] = {0};
  preds[2] = {1};

  std::vector<std::vector<uint32_t>> result;

  net.reconstruct_paths(source, targets, preds, result);

  ASSERT_EQ(result.size(), 1);
  EXPECT_EQ(result[0], (std::vector<uint32_t>{0, 1, 2}));
}

TEST(TestNetwork, ReconstructPathsEmptyTargets) {
  NetworkProtected net;

  uint32_t source = 0;
  std::unordered_set<uint32_t> targets;

  std::vector<std::vector<uint32_t>> preds;

  std::vector<std::vector<uint32_t>> result;

  net.reconstruct_paths(source, targets, preds, result);

  EXPECT_TRUE(result.empty());
}

TEST(TestNetwork, BFS_CoreSimpleLinearPath) {
  NetworkProtected net;
  net.add_nodes({"A","B","C"});
  net.add_edge("A","B");
  net.add_edge("B","C");

  std::unordered_set<uint32_t> targets = {2};
  auto paths = net.find_all_shortest_paths_bfs_core(0, targets);

  ASSERT_EQ(paths.size(), 1);
  EXPECT_EQ(paths[0], (std::vector<uint32_t>{0,1,2}));
}

TEST(TestNetwork, BFS_CoreMultipleShortestPaths) {
  NetworkProtected net;
  net.add_nodes({"A","B","C","D"});
  net.add_edge("A","B");
  net.add_edge("A","C");
  net.add_edge("B","D");
  net.add_edge("C","D");

  std::unordered_set<uint32_t> targets = {3};
  auto paths = net.find_all_shortest_paths_bfs_core(0, targets);

  EXPECT_EQ(paths.size(), 2);

  std::vector<std::vector<uint32_t>> expected = {
    {0,1,3},
    {0,2,3}
  };

  std::sort(paths.begin(), paths.end());
  std::sort(expected.begin(), expected.end());

  EXPECT_EQ(paths, expected);
}

TEST(TestNetwork, BFS_CoreEarlyStopping) {
  NetworkProtected net;
  net.add_nodes({"A","B","C","D","E","F","G","H"});
  net.add_edge("A","B");
  net.add_edge("B","C");
  net.add_edge("A","F");
  net.add_edge("F","G");
  net.add_edge("G","H");

  std::unordered_set<uint32_t> targets = {2};
  auto paths = net.find_all_shortest_paths_bfs_core(0, targets);

  ASSERT_EQ(paths.size(), 1);
  EXPECT_EQ(paths[0], (std::vector<uint32_t>{0,1,2}));
}

TEST(TestNetwork, BFS_CoreNoTargetsReturnsAllReachablePaths) {
  NetworkProtected net;
  net.add_nodes({"A","B","C","D"});
  net.add_edge("A","B");
  net.add_edge("A","C");
  net.add_edge("B","D");

  std::unordered_set<uint32_t> targets; // empty
  auto paths = net.find_all_shortest_paths_bfs_core(0, targets);

  // Should return shortest paths to 1, 2, 3
  ASSERT_EQ(paths.size(), 3);

  std::vector<std::vector<uint32_t>> expected = {
    {0,1},
    {0,2},
    {0,1,3}
  };

  std::sort(paths.begin(), paths.end());
  std::sort(expected.begin(), expected.end());

  EXPECT_EQ(paths, expected);
}

TEST(TestNetwork, BFS_CoreUnreachableTarget) {
  NetworkProtected net;
  net.add_nodes({"A","B","C","D"});
  net.add_edge("A","B");
  net.add_edge("B","C");

  std::unordered_set<uint32_t> targets = {3}; // unreachable
  auto paths = net.find_all_shortest_paths_bfs_core(0, targets);

  EXPECT_TRUE(paths.empty());
}

TEST(TestNetwork, BFS_CoreSourceInTargetsIsRemoved) {
  NetworkProtected net;
  net.add_nodes({"A","B","C","D"});
  net.add_edge("A","B");

  // target contains source
  std::unordered_set<uint32_t> targets = {0, 1};

  auto paths = net.find_all_shortest_paths_bfs_core(0, targets);

  ASSERT_EQ(paths.size(), 1);
  std::vector<std::vector<uint32_t>> expected = {
    {0,1}
  };

  std::sort(paths.begin(), paths.end());
  std::sort(expected.begin(), expected.end());

  EXPECT_EQ(paths, expected);
}

TEST(TestNetwork, BFS_CoreLongPaths) {
  NetworkProtected net;
  net.add_nodes({"A","B","C","D","E","F"});
  net.add_edge("A","B");
  net.add_edge("A","C");
  net.add_edge("B","D");
  net.add_edge("C","E");
  net.add_edge("D","F");
  net.add_edge("E","F");

  std::unordered_set<uint32_t> targets = {5};
  auto paths = net.find_all_shortest_paths_bfs_core(0, targets);

  ASSERT_EQ(paths.size(), 2);

  std::vector<std::vector<uint32_t>> expected = {
    {0,1,3,5},
    {0,2,4,5}
  };

  std::sort(paths.begin(), paths.end());
  std::sort(expected.begin(), expected.end());
  EXPECT_EQ(paths, expected);
}

TEST(TestNetwork, DijkstraCoreSinglePath) {
  NetworkProtected net;
  net.add_nodes({"0","1","2"});
  net.add_edge("0","1",1.0);
  net.add_edge("1","2",2.0);
  
  uint32_t source = 0;
  std::unordered_set<uint32_t> targets = {2};

  auto paths = net.find_all_shortest_paths_dijkstra_core(source, targets);

  ASSERT_EQ(paths.size(), 1);
  EXPECT_EQ(paths[0], (std::vector<uint32_t>{0, 1, 2}));
}

TEST(TestNetwork, DijkstraCoreMultiplePaths) {
  NetworkProtected net;
  net.add_nodes({"0","1","2","3"});
  net.add_edge("0","1",1.0);
  net.add_edge("0","2",1.0);
  net.add_edge("1","3",1.0);
  net.add_edge("2","3",1.0);

  uint32_t source = 0;
  std::unordered_set<uint32_t> targets = {3};

  auto paths = net.find_all_shortest_paths_dijkstra_core(source, targets);

  ASSERT_EQ(paths.size(), 2);

  std::vector<std::vector<uint32_t>> expected = {
    {0, 1, 3},
    {0, 2, 3}
  };

  auto sort_paths = [](auto& paths) { std::sort(paths.begin(), paths.end()); };
  sort_paths(paths);
  sort_paths(expected);

  EXPECT_EQ(paths, expected);
}

TEST(TestNetwork, DijkstraCoreMultipleTargets) {
  NetworkProtected net;
  net.add_nodes({"0","1","2","3"});
  net.add_edge("0","1",1.0);
  net.add_edge("1","2",1.0);

  uint32_t source = 0;
  std::unordered_set<uint32_t> targets = {2, 3};

  auto paths = net.find_all_shortest_paths_dijkstra_core(source, targets);

  // Only target 2 is reachable
  ASSERT_EQ(paths.size(), 1);
  EXPECT_EQ(paths[0], (std::vector<uint32_t>{0, 1, 2}));
}

TEST(TestNetwork, DijkstraCoreAllReachableNodes) {
  NetworkProtected net;
  net.add_nodes({"0","1","2"});
  net.add_edge("0","1",1.0);
  net.add_edge("1","2",1.0);

  uint32_t source = 0;
  std::unordered_set<uint32_t> targets; // empty

  auto paths = net.find_all_shortest_paths_dijkstra_core(source, targets);

  std::vector<std::vector<uint32_t>> expected = {
    {0, 1},
    {0, 1, 2}
  };

  auto sort_paths = [](auto& paths) { std::sort(paths.begin(), paths.end()); };
  sort_paths(paths);
  sort_paths(expected);

  EXPECT_EQ(paths, expected);
}

TEST(TestNetwork, DijkstraCoreWeightedEdges) {
  NetworkProtected net;
  net.add_nodes({"0","1","2","3","4","5"});
  net.add_edge("0","1",1.0);
  net.add_edge("0","2",2.0);
  net.add_edge("1","3",5.0);
  net.add_edge("2","3",1.0);

  uint32_t source = 0;
  std::unordered_set<uint32_t> targets = {3};

  auto paths = net.find_all_shortest_paths_dijkstra_core(source, targets);

  // Shortest path is 0 -> 2 -> 3
  ASSERT_EQ(paths.size(), 1);
  EXPECT_EQ(paths[0], (std::vector<uint32_t>{0, 2, 3}));
}

TEST(TestNetwork, DijkstraCoreEarlyStopping) {
  NetworkProtected net;
  net.add_nodes({"0","1","2","3","4","5"});
  net.add_edge("0","1",1.0);
  net.add_edge("0","2",1.0);
  net.add_edge("1","3",1.0);
  net.add_edge("2","4",1.0);
  net.add_edge("3","5",1.0);
  net.add_edge("4","5",2.0);

  uint32_t source = 0;
  std::unordered_set<uint32_t> targets = {3, 4};

  auto paths = net.find_all_shortest_paths_dijkstra_core(source, targets);

  // Only targets 3 and 4 should be returned, node 5 is ignored
  ASSERT_EQ(paths.size(), 2);

  std::vector<std::vector<uint32_t>> expected = {
    {0, 1, 3},
    {0, 2, 4}
  };

  auto sort_paths = [](auto& paths) { std::sort(paths.begin(), paths.end()); };
  sort_paths(paths);
  sort_paths(expected);

  EXPECT_EQ(paths, expected);
}

TEST(TestNetwork, FindAllShortestPathsBfsMocked) {
  NetworkMockShortestPathSingle net;
  net.add_nodes({"A","B","C"});
  net.add_edge("A","B");
  net.add_edge("B","C");

  std::unordered_set<uint32_t> expected_targets;
  expected_targets.insert(2);

  // Mock the BFS core method to return a fixed index path
  EXPECT_CALL(net, find_all_shortest_paths_bfs_core(0, expected_targets))
    .WillOnce(testing::Return(std::vector<std::vector<uint32_t>>{{0, 42}}));

  // Mock the conversion method to return labels for the mocked path
  EXPECT_CALL(net, convert_paths_to_labels(testing::_))
    .WillOnce(testing::Return(std::vector<std::vector<std::string>>{{"SOURCE", "TARGET"}}));

  std::unordered_set<std::string> targets = {"C"};
  auto paths = net.find_all_shortest_paths_bfs("A", targets);

  ASSERT_EQ(paths.size(), 1);
  EXPECT_EQ(paths[0], (std::vector<std::string>{"SOURCE", "TARGET"}));
}

TEST(TestNetwork, FindAllShortestPathsBfsMockedRemovesDuplicateTargets) {
  NetworkMockShortestPathSingle net;
  net.add_nodes({"A","B","C"});
  net.add_edge("A","B");
  net.add_edge("B","C");

  std::unordered_set<uint32_t> expected_targets;
  expected_targets.insert(2);

  // Mock the BFS core method to return a fixed index path
  EXPECT_CALL(net, find_all_shortest_paths_bfs_core(0, expected_targets))
    .WillOnce(testing::Return(std::vector<std::vector<uint32_t>>{{0, 42}}));

  // Mock the conversion method to return labels for the mocked path
  EXPECT_CALL(net, convert_paths_to_labels(testing::_))
    .WillOnce(testing::Return(std::vector<std::vector<std::string>>{{"SOURCE", "TARGET"}}));

  std::unordered_set<std::string> targets = {"C", "C"};
  auto paths = net.find_all_shortest_paths_bfs("A", targets);

  ASSERT_EQ(paths.size(), 1);
  EXPECT_EQ(paths[0], (std::vector<std::string>{"SOURCE", "TARGET"}));
}

TEST(TestNetwork, FindAllShortestPathsBfsThrowsOnSourceNotFound) {
  Network net;
  net.add_nodes({"A","B","C"});
  net.add_edge("A","B");
  net.add_edge("B","C");

  std::unordered_set<std::string> targets = {"C"};
  
  ASSERT_THAT(
    [&](){auto paths = net.find_all_shortest_paths_bfs("D", targets); },
    testing::ThrowsMessage<std::invalid_argument>("Network::find_all_shortest_paths_bfs - Source node label 'D' not found in the network.")
  );
}

TEST(TestNetwork, FindAllShortestPathsBfsThrowsOnTargetNotFound) {
  Network net;
  net.add_nodes({"A","B","C"});
  net.add_edge("A","B");
  net.add_edge("B","C");

  std::unordered_set<std::string> targets = {"D"};
  
  ASSERT_THAT(
    [&](){auto paths = net.find_all_shortest_paths_bfs("A", targets); },
    testing::ThrowsMessage<std::invalid_argument>("Network::find_all_shortest_paths_bfs - Target node label 'D' not found in the network.")
  );
}

TEST(TestNetwork, FindAllShortestPathsDijkstraMocked) {
  NetworkMockShortestPathSingle net;
  net.add_nodes({"A","B","C"});
  net.add_edge("A","B");
  net.add_edge("B","C");

  std::unordered_set<uint32_t> expected_targets;
  expected_targets.insert(2);

  // Mock the BFS core method to return a fixed index path
  EXPECT_CALL(net, find_all_shortest_paths_dijkstra_core(0, expected_targets))
    .WillOnce(testing::Return(std::vector<std::vector<uint32_t>>{{0, 42}}));

  // Mock the conversion method to return labels for the mocked path
  EXPECT_CALL(net, convert_paths_to_labels(testing::_))
    .WillOnce(testing::Return(std::vector<std::vector<std::string>>{{"SOURCE", "TARGET"}}));

  std::unordered_set<std::string> targets = {"C"};
  auto paths = net.find_all_shortest_paths_dijkstra("A", targets);

  ASSERT_EQ(paths.size(), 1);
  EXPECT_EQ(paths[0], (std::vector<std::string>{"SOURCE", "TARGET"}));
}

TEST(TestNetwork, FindAllShortestPathsDijkstraMockedRemovesDuplicateTargets) {
  NetworkMockShortestPathSingle net;
  net.add_nodes({"A","B","C"});
  net.add_edge("A","B");
  net.add_edge("B","C");

  std::unordered_set<uint32_t> expected_targets;
  expected_targets.insert(2);

  // Mock the BFS core method to return a fixed index path
  EXPECT_CALL(net, find_all_shortest_paths_dijkstra_core(0, expected_targets))
    .WillOnce(testing::Return(std::vector<std::vector<uint32_t>>{{0, 42}}));

  // Mock the conversion method to return labels for the mocked path
  EXPECT_CALL(net, convert_paths_to_labels(testing::_))
    .WillOnce(testing::Return(std::vector<std::vector<std::string>>{{"SOURCE", "TARGET"}}));

  std::unordered_set<std::string> targets = {"C", "C"};
  auto paths = net.find_all_shortest_paths_dijkstra("A", targets);

  ASSERT_EQ(paths.size(), 1);
  EXPECT_EQ(paths[0], (std::vector<std::string>{"SOURCE", "TARGET"}));
}

TEST(TestNetwork, FindAllShortestPathsDijkstraThrowsOnSourceNotFound) {
  Network net;
  net.add_nodes({"A","B","C"});
  net.add_edge("A","B");
  net.add_edge("B","C");

  std::unordered_set<std::string> targets = {"C"};
  
  ASSERT_THAT(
    [&](){auto paths = net.find_all_shortest_paths_dijkstra("D", targets); },
    testing::ThrowsMessage<std::invalid_argument>("Network::find_all_shortest_paths_dijkstra - Source node label 'D' not found in the network.")
  );
}

TEST(TestNetwork, FindAllShortestPathsDijkstraThrowsOnTargetNotFound) {
  Network net;
  net.add_nodes({"A","B","C"});
  net.add_edge("A","B");
  net.add_edge("B","C");

  std::unordered_set<std::string> targets = {"D"};

  ASSERT_THAT(
    [&](){auto paths = net.find_all_shortest_paths_dijkstra("A", targets); },
    testing::ThrowsMessage<std::invalid_argument>("Network::find_all_shortest_paths_dijkstra - Target node label 'D' not found in the network.")
  );
}

TEST(TestNetwork, FindAllShortestPathsBfsMultiThrowsOnSourceNotFound) {
  Network net;
  net.add_nodes({"A","B","C"});
  net.add_edge("A","B");
  net.add_edge("B","C");

  std::unordered_set<std::string> sources = {"A", "D"};
  std::unordered_set<std::string> targets = {"C"};

  ASSERT_THAT(
    [&](){auto paths = net.find_all_shortest_paths_bfs(sources, targets); },
    testing::ThrowsMessage<std::invalid_argument>("Network::find_all_shortest_paths_bfs - Source node label 'D' not found in the network.")
  );
}

TEST(TestNetwork, FindAllShortestPathsBfsMultiThrowsOnTargetNotFound) {
  Network net;
  net.add_nodes({"A","B","C"});
  net.add_edge("A","B");
  net.add_edge("B","C");

  std::unordered_set<std::string> sources = {"A", "B"};
  std::unordered_set<std::string> targets = {"C", "D"};

  ASSERT_THAT(
    [&](){auto paths = net.find_all_shortest_paths_bfs(sources, targets); },
    testing::ThrowsMessage<std::invalid_argument>("Network::find_all_shortest_paths_bfs - Target node label 'D' not found in the network.")
  );
}

TEST(TestNetwork, FindAllShortestPathsBfsMultiEmptySources) {
  Network net;
  net.add_nodes({"A","B","C"});
  net.add_edge("A","B");
  net.add_edge("B","C");

  // Empty sources
  std::unordered_set<std::string> sources1;
  std::unordered_set<std::string> targets1 = {"B"};
  auto paths_empty_sources = net.find_all_shortest_paths_bfs(sources1, targets1);
  EXPECT_TRUE(paths_empty_sources.empty());
}

TEST(TestNetwork, FindAllShortestPathsBfsMultiEmptyTargets) {
  Network net;
  net.add_nodes({"A","B","C"});
  net.add_edge("A","B");
  net.add_edge("B","C");

  std::unordered_set<std::string> sources = {"A"};
  std::unordered_set<std::string> targets; // EMPTY

  auto paths = net.find_all_shortest_paths_bfs(sources, targets);

  // From A in this graph:
  // A→B
  // A→B→C
  ASSERT_EQ(paths.size(), 2);

  EXPECT_THAT(paths, ::testing::UnorderedElementsAre(
    std::vector<std::string>{"A","B"},
    std::vector<std::string>{"A","B","C"}
  ));
}

TEST(TestNetwork, FindAllShortestPathsBfsMultiNoReachableTargets) {
  Network net;
  net.add_nodes({"A","B","C","X","Y"});
  net.add_edge("A","B");
  net.add_edge("B","C");
  net.add_edge("X","Y"); // separate component

  std::unordered_set<std::string> sources = {"A"};
  std::unordered_set<std::string> targets = {"X"};

  auto paths = net.find_all_shortest_paths_bfs(sources, targets);

  EXPECT_TRUE(paths.empty());
}


TEST(TestNetwork, FindAllShortestPathsBfsMultiSingleSourceSingleTarget) {
  NetworkMockShortestPathSingle net;
  net.add_nodes({"A","B","C"});
  net.add_edge("A","B");
  net.add_edge("B","C");

  std::vector<std::vector<uint32_t>> mock_index_paths = {{0, 1}};
  std::vector<std::vector<std::string>> mock_label_paths = {{"A", "B"}};

  // Mock the BFS core method to return a fixed index path
  EXPECT_CALL(net, find_all_shortest_paths_bfs_core(0, ::testing::ContainerEq(std::unordered_set<uint32_t>{1})))
    .WillOnce(testing::Return(mock_index_paths));

  // Mock the conversion method to return labels for the mocked path
  EXPECT_CALL(net, convert_paths_to_labels(mock_index_paths))
    .WillOnce(testing::Return(mock_label_paths));

  std::unordered_set<std::string> sources = {"A"};
  std::unordered_set<std::string> targets = {"B"};
  auto paths = net.find_all_shortest_paths_bfs(sources, targets);

  ASSERT_EQ(paths.size(), 1);
  EXPECT_EQ(paths[0], (std::vector<std::string>{"A", "B"}));
}

TEST(TestNetwork, FindAllShortestPathsBfsMultiMultipleSourcesMultipleTargets) {
  NetworkMockShortestPathSingle net;
  net.add_nodes({"A","B","C","D"});
  net.add_edge("A","B");
  net.add_edge("B","C");

  std::unordered_set<uint32_t> targets_idx = {1, 2};
  std::vector<std::vector<uint32_t>> mock_index_paths_A = {{0, 1}, {0, 2}};
  std::vector<std::vector<uint32_t>> mock_index_paths_B = {{3, 1}, {3, 2}};

  // BFS core called for each source
  EXPECT_CALL(net, find_all_shortest_paths_bfs_core(0, ::testing::ContainerEq(targets_idx)))
    .WillOnce(::testing::Return(mock_index_paths_A));

  EXPECT_CALL(net, find_all_shortest_paths_bfs_core(3, ::testing::ContainerEq(targets_idx)))
    .WillOnce(::testing::Return(mock_index_paths_B));

  // Conversion to labels
  EXPECT_CALL(net, convert_paths_to_labels(mock_index_paths_A))
    .WillOnce(::testing::Return(std::vector<std::vector<std::string>>{{"A", "B"}, {"A", "C"}}));

  EXPECT_CALL(net, convert_paths_to_labels(mock_index_paths_B))
    .WillOnce(::testing::Return(std::vector<std::vector<std::string>>{{"D", "B"}, {"D", "C"}}));

  std::unordered_set<std::string> sources = {"A", "D"};
  std::unordered_set<std::string> targets = {"B", "C"};
  auto paths = net.find_all_shortest_paths_bfs(sources, targets);

  // There should be 4 paths in total
  ASSERT_EQ(paths.size(), 4);
  std::vector<std::vector<std::string>> expected = {
    {"A", "B"}, {"A", "C"}, {"D", "B"}, {"D", "C"}
  };

  std::sort(paths.begin(), paths.end());

  EXPECT_EQ(paths, expected);
}

TEST(TestNetwork, FindAllShortestPathsBfsMultiMultipleSourcesMultipleTargetsParallelMerge) {
  Network net;
  net.add_nodes({"A","B","C","D"});
  net.add_edge("A", "B"); 
  net.add_edge("A", "C");
  net.add_edge("B", "C");
  net.add_edge("C", "D");

  // Sources and targets
  std::unordered_set<std::string> sources = {"A", "B", "C", "D"};
  std::unordered_set<std::string> targets = {"A", "B", "C", "D"};

  #ifdef USE_OPENMP
  // Force multiple threads only when OpenMP is available
  omp_set_num_threads(4);
  #endif

  auto paths = net.find_all_shortest_paths_bfs(sources, targets);

  // Expected number of paths = sources.size() * targets.size()
  ASSERT_EQ(paths.size(), 12);

  // Construct expected paths
  std::vector<std::vector<std::string>> expected;
  expected.push_back({"A", "B"});
  expected.push_back({"A", "C"});
  expected.push_back({"A", "C", "D"});
  expected.push_back({"B", "A"});
  expected.push_back({"B", "C"});
  expected.push_back({"B", "C", "D"});
  expected.push_back({"C", "A"});
  expected.push_back({"C", "B"});
  expected.push_back({"C", "D"});
  expected.push_back({"D", "C"});
  expected.push_back({"D", "C", "A"});
  expected.push_back({"D", "C", "B"});

  std::sort(paths.begin(), paths.end());

  EXPECT_EQ(paths, expected);
}

TEST(TestNetwork, FindAllShortestPathsDijkstraMultiThrowsOnSourceNotFound) {
  Network net;
  net.add_nodes({"A","B","C"});
  net.add_edge("A","B");
  net.add_edge("B","C");

  std::unordered_set<std::string> sources = {"A", "D"};
  std::unordered_set<std::string> targets = {"C"};

  ASSERT_THAT(
    [&](){ auto paths = net.find_all_shortest_paths_dijkstra(sources, targets); },
    testing::ThrowsMessage<std::invalid_argument>(
      "Network::find_all_shortest_paths_dijkstra - Source node label 'D' not found in the network.")
  );
}

TEST(TestNetwork, FindAllShortestPathsDijkstraMultiThrowsOnTargetNotFound) {
  Network net;
  net.add_nodes({"A","B","C"});
  net.add_edge("A","B");
  net.add_edge("B","C");

  std::unordered_set<std::string> sources = {"A","B"};
  std::unordered_set<std::string> targets = {"C","D"};

  ASSERT_THAT(
    [&](){ auto paths = net.find_all_shortest_paths_dijkstra(sources, targets); },
    testing::ThrowsMessage<std::invalid_argument>(
      "Network::find_all_shortest_paths_dijkstra - Target node label 'D' not found in the network.")
  );
}

TEST(TestNetwork, FindAllShortestPathsDijkstraMultiEmptySources) {
  Network net;
  net.add_nodes({"A","B","C"});
  net.add_edge("A","B");
  net.add_edge("B","C");

  std::unordered_set<std::string> sources;  // empty
  std::unordered_set<std::string> targets = {"B"};

  auto paths = net.find_all_shortest_paths_dijkstra(sources, targets);
  EXPECT_TRUE(paths.empty());
}

TEST(TestNetwork, FindAllShortestPathsDijkstraMultiEmptyTargets) {
  Network net;
  net.add_nodes({"A","B","C"});
  net.add_edge("A","B", 1.0);
  net.add_edge("B","C", 1.0);

  std::unordered_set<std::string> sources = {"A"};
  std::unordered_set<std::string> targets; // empty

  auto paths = net.find_all_shortest_paths_dijkstra(sources, targets);

  // Expected: A->B and A->B->C
  ASSERT_EQ(paths.size(), 2);
  EXPECT_THAT(paths, ::testing::UnorderedElementsAre(
    std::vector<std::string>{"A","B"},
    std::vector<std::string>{"A","B","C"}
  ));
}

TEST(TestNetwork, FindAllShortestPathsDijkstraMultiNoReachableTargets) {
  Network net;
  net.add_nodes({"A","B","C","X","Y"});
  net.add_edge("A","B", 1.0);
  net.add_edge("B","C", 1.0);
  net.add_edge("X","Y", 1.0); // separate component

  std::unordered_set<std::string> sources = {"A"};
  std::unordered_set<std::string> targets = {"X"};

  auto paths = net.find_all_shortest_paths_dijkstra(sources, targets);
  EXPECT_TRUE(paths.empty());
}

TEST(TestNetwork, FindAllShortestPathsDijkstraMultiSingleSourceSingleTarget) {
  NetworkMockShortestPathSingle net;
  net.add_nodes({"A","B","C"});
  net.add_edge("A","B");
  net.add_edge("B","C");

  std::vector<std::vector<uint32_t>> mock_index_paths = {{0, 1}};
  std::vector<std::vector<std::string>> mock_label_paths = {{"A","B"}};

  EXPECT_CALL(
    net,
    find_all_shortest_paths_dijkstra_core(
      0, ::testing::ContainerEq(std::unordered_set<uint32_t>{1})
    )
  ).WillOnce(::testing::Return(mock_index_paths));

  EXPECT_CALL(net, convert_paths_to_labels(mock_index_paths))
    .WillOnce(::testing::Return(mock_label_paths));

  std::unordered_set<std::string> sources = {"A"};
  std::unordered_set<std::string> targets = {"B"};

  auto paths = net.find_all_shortest_paths_dijkstra(sources, targets);

  ASSERT_EQ(paths.size(), 1);
  EXPECT_EQ(paths[0], (std::vector<std::string>{"A","B"}));
}

TEST(TestNetwork, FindAllShortestPathsDijkstraMultiMultipleSourcesMultipleTargets) {
  NetworkMockShortestPathSingle net;
  net.add_nodes({"A","B","C","D"});

  std::unordered_set<uint32_t> targets_idx = {1, 2};
  std::vector<std::vector<uint32_t>> mock_index_paths_A = {{0,1},{0,2}};
  std::vector<std::vector<uint32_t>> mock_index_paths_D = {{3,1},{3,2}};

  EXPECT_CALL(net, find_all_shortest_paths_dijkstra_core(0, targets_idx))
    .WillOnce(::testing::Return(mock_index_paths_A));
  EXPECT_CALL(net, find_all_shortest_paths_dijkstra_core(3, targets_idx))
    .WillOnce(::testing::Return(mock_index_paths_D));

  EXPECT_CALL(net, convert_paths_to_labels(mock_index_paths_A))
    .WillOnce(::testing::Return(std::vector<std::vector<std::string>>{
      {"A","B"}, {"A","C"}
    }));
  EXPECT_CALL(net, convert_paths_to_labels(mock_index_paths_D))
    .WillOnce(::testing::Return(std::vector<std::vector<std::string>>{
      {"D","B"}, {"D","C"}
    }));

  std::unordered_set<std::string> sources = {"A","D"};
  std::unordered_set<std::string> targets = {"B","C"};

  auto paths = net.find_all_shortest_paths_dijkstra(sources, targets);

  ASSERT_EQ(paths.size(), 4);
  std::vector<std::vector<std::string>> expected = {
    {"A","B"}, {"A","C"}, {"D","B"}, {"D","C"}
  };

  std::sort(paths.begin(), paths.end());
  EXPECT_EQ(paths, expected);
}

TEST(TestNetwork, FindAllShortestPathsDijkstraMultiMultipleSourcesMultipleTargetsParallelMerge) {
  Network net;
  net.add_nodes({"A","B","C","D"});
  net.add_edge("A","B", 1.0); 
  net.add_edge("A","C", 1.0);
  net.add_edge("B","C", 1.0);
  net.add_edge("C","D", 1.0);

  std::unordered_set<std::string> sources = {"A","B","C","D"};
  std::unordered_set<std::string> targets = {"A","B","C","D"};

  #ifdef USE_OPENMP
  omp_set_num_threads(4);
  #endif

  auto paths = net.find_all_shortest_paths_dijkstra(sources, targets);

  ASSERT_EQ(paths.size(), 12);

  std::vector<std::vector<std::string>> expected {
    {"A","B"},
    {"A","C"},
    {"A","C","D"},
    {"B","A"},
    {"B","C"},
    {"B","C","D"},
    {"C","A"},
    {"C","B"},
    {"C","D"},
    {"D","C"},
    {"D","C","A"},
    {"D","C","B"}
  };

  std::sort(paths.begin(), paths.end());
  EXPECT_EQ(paths, expected);
}

TEST(TestNetwork, FindAllShortestPathsHybridThrowsOnSourceNotFound) {
  Network net;
  net.add_nodes({"A"});

  std::unordered_set<std::string> S = {"B"};
  std::unordered_set<std::string> T = {"A"};

  ASSERT_THAT(
    [&](){ auto paths = net.find_all_shortest_paths(S, T); },
    testing::ThrowsMessage<std::invalid_argument>(
      "Network::find_all_shortest_paths - Source node label 'B' not found in the network.")
  );
}

TEST(TestNetwork, FindAllShortestPathsHybridThrowsOnTargetNotFound) {
  Network net;
  net.add_nodes({"A"});

  std::unordered_set<std::string> S = {"A"};
  std::unordered_set<std::string> T = {"B"};

  ASSERT_THAT(
    [&](){ auto paths = net.find_all_shortest_paths(S, T); },
    testing::ThrowsMessage<std::invalid_argument>(
      "Network::find_all_shortest_paths - Target node label 'B' not found in the network.")
  );
}

TEST(TestNetwork, FindAllShortestPathsHybridUsesBfsWhenUseWeightsFalse) {
  NetworkMockShortestPath net;
  net.add_nodes({"A","B"});
  net.add_edge("A","B");

  EXPECT_CALL(net, find_all_shortest_paths_bfs(::testing::_, ::testing::_))
    .Times(1);
  EXPECT_CALL(net, find_all_shortest_paths_dijkstra(::testing::_, ::testing::_))
    .Times(0);

  std::unordered_set<std::string> S = {"A"};
  std::unordered_set<std::string> T = {"B"};
  net.find_all_shortest_paths(S, T, false);
}

TEST(TestNetwork, FindAllShortestPathsHybridUsesBfsOnEqualWeights) {
  NetworkMockShortestPath net;
  net.add_nodes({"A","B","C"});
  net.add_edge("A","B", 1.3);
  net.add_edge("A","C", 1.3);

  EXPECT_CALL(net, find_all_shortest_paths_bfs(::testing::_, ::testing::_))
    .Times(1);
  EXPECT_CALL(net, find_all_shortest_paths_dijkstra(::testing::_, ::testing::_))
    .Times(0);

  std::unordered_set<std::string> S = {"A"};
  std::unordered_set<std::string> T = {"B"};
  net.find_all_shortest_paths(S, T, false);
}

TEST(TestNetwork, FindAllShortestPathsHybridUsesDijkstraOnDifferentWeights) {
  NetworkMockShortestPath net;
  net.add_nodes({"A","B","C"});
  net.add_edge("A","B", 1.0);
  net.add_edge("A","C", 2.0);

  EXPECT_CALL(net, find_all_shortest_paths_bfs(::testing::_, ::testing::_))
    .Times(0);
  EXPECT_CALL(net, find_all_shortest_paths_dijkstra(::testing::_, ::testing::_))
    .Times(1);

  std::unordered_set<std::string> S = {"A"};
  std::unordered_set<std::string> T = {"B"};
  net.find_all_shortest_paths(S, T, true);
}

TEST(TestNetwork, FindAllShortestPathsHybridReturnsEarlyOnEmptySource) {
  NetworkMockShortestPath net;
  net.add_nodes({"A","B","C"});
  net.add_edge("A","B", 1.0);
  net.add_edge("A","C", 2.0);

  EXPECT_CALL(net, find_all_shortest_paths_bfs(::testing::_, ::testing::_))
    .Times(0);
  EXPECT_CALL(net, find_all_shortest_paths_dijkstra(::testing::_, ::testing::_))
    .Times(0);

  std::unordered_set<std::string> S;
  std::unordered_set<std::string> T = {"B"};
  net.find_all_shortest_paths(S, T, true);
}

TEST(TestNetwork, FindAllShortestPathsHybridReturnsEmptyIfNoEdges) {
  Network net;
  net.add_nodes({"A","B","C"});

  std::unordered_set<std::string> S = {"A"};
  std::unordered_set<std::string> T = {"B"};
  auto paths = net.find_all_shortest_paths(S, T, true);

  EXPECT_TRUE(paths.empty());
}

TEST(TestNetwork, HybridSingleSourceCallsMultiSourceVersion) {
  NetworkMockSingleToMulti net;
  net.add_nodes({"A", "B"});
  net.add_edge("A", "B");

  std::unordered_set<std::string> expected_S = {"A"};
  std::unordered_set<std::string> expected_T = {"B"};
  bool expected_use_weights = true;

  // What the mocked multi-source method should return
  std::vector<std::vector<std::string>> mock_return = { {"A", "B"} };

  // Expect multi-source version to be called once with correct args
  EXPECT_CALL(net,
              find_all_shortest_paths(
                  testing::ContainerEq(expected_S),
                  testing::ContainerEq(expected_T),
                  expected_use_weights))
      .Times(1)
      .WillOnce(testing::Return(mock_return));

  // Call the single-source version
  auto result = net.find_all_shortest_paths(expected_S, expected_T, true);

  // Verify the returned value is the mock’s value
  ASSERT_EQ(result.size(), 1);
  EXPECT_EQ(result[0], (std::vector<std::string>{"A", "B"}));
}
