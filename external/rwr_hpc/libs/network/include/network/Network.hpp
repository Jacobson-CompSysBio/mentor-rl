/// @file Network.hpp
/// @brief Defines the Network class for representing and manipulating graph structures,
///        along with the Edge struct used for adjacency representation.
/// 
/// This file provides a lightweight graph abstraction that supports undirected and directed graphs,
/// multigraph support, node label indexing, and efficient CSR matrix generation for sparse matrix applications.
/// 
/// Typical use cases include graph analysis, transition matrix construction for random walks,
/// and integration into multilayer or multiplex network models.
/// 
/// @author Ken Smith
/// @date 2025-07-24

#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include <sparse/CSR_Matrix.hpp>

/// @struct Edge
/// @brief Represents a single edge in the adjacency list of a node.
/// 
/// Each edge contains the index of the target node and the associated edge weight.
/// This struct is used within the Network class to store adjacency lists efficiently.
struct Edge { 
  uint32_t target = 0; ///< Index of the target node.
  double weight = 0.0;    ///< Weight of the edge (must be non-zero).
};

enum class MergeMethod {
  Max,
  Min,
  All,
  Sum,
  Mean
};

/// @class Network
/// @brief A labeled graph class supporting directed/undirected and multigraph configurations.
/// 
/// The Network class allows flexible construction and manipulation of graphs with string-labeled nodes.
/// It supports adding and querying edges and nodes, converting to CSR matrices, and computing degree statistics.
/// 
/// The internal structure uses adjacency lists and bidirectional label mappings for efficient lookup.
class Network {
public:
  friend class Multiplex;

  /// @brief Default constructor for Network.
  ///
  /// @details Initializes an empty network with no nodes, edges, or labels.
  Network() = default;

  /// @brief Destructor for Network.
  ///
  /// @details Cleans up internal resources. Default implementation is sufficient.
  ~Network() = default;

  /// @brief Get the number of nodes in the network.
  inline std::size_t get_n_nodes() const { return labels_.size(); }

  /// @brief Get a sorted list of all node labels in the network.
  const std::vector<std::string>& get_labels() const { return index_to_label_; };

  /// @brief Get the index corresponding to a node label.
  ///
  /// @param label The label to lookup.
  ///
  /// @return The index of the node.
  ///
  /// @throws std::runtime_error if label is not found.
  virtual uint32_t get_idx_of_label(const std::string &label) const;

  /// @brief Get the label corresponding to a node index.
  ///
  /// @param index The node index.
  ///
  /// @return The label of the node.
  ///
  /// @throws std::out_of_range if index is invalid.
  inline std::string get_label_of_index(std::size_t index) const {return index_to_label_.at(index); }

  /// @brief Get the number of edges in the network.
  //
  /// @return If undirected, returns nnz/2; if directed, returns nnz.
  inline std::size_t get_n_edges() const {return directed_ ? nnz_ : nnz_ / 2; }

  /// @brief Get the number of non-zero edges, where the source and target of the edge is in node_label_list
  ///
  /// @param node_label_list Optional list of labels to restrict the count (default: all).
  ///
  /// @return Number of non-zero entries involving the specified nodes.
  std::size_t get_nnz(const std::vector<std::string>& node_label_list = {}) const;

  /// @brief Check if the network is directed.
  inline bool is_directed() const {return directed_; }

  /// @brief Check if the network is a multigraph.
  inline bool is_multigraph() const {return multigraph_; }

  /// @brief Get the degree of a node (undirected networks only).
  ///
  /// @param idx Index of the node.
  ///
  /// @return Number of incident edges.
  ///
  /// @throws std::out_of_range if idx is invalid.
  /// @throws std::runtime_error if the network is directed.
  virtual std::size_t degree(const std::size_t idx) const;

  /// @brief Get the maximum node degree in the network.
  inline std::size_t get_max_degree() const {return max_degree_; }

  /// @brief Check whether a node with a given label exists in the network.
  ///
  /// @param label The label to check for.
  ///
  /// @return true if found, false otherwise.
  inline bool contains_node(const std::string& label) const { return (labels_.find(label) != labels_.end()); }
  
  /// @brief Add a node with the specified label.
  ///
  /// @param label The label of the node.
  ///
  /// @return The index of the node (existing or new).
  virtual std::size_t add_node(const std::string &label);

  /// @brief Add multiple nodes to the network.
  ///
  /// @param labels A list of labels to add.
  void add_nodes(const std::vector<std::string> &labels);

  /// @brief Add an edge between labeled nodes, creating missing nodes if allowed.
  ///
  /// @param src Source node label.
  /// @param tgt Target node label.
  /// @param weight Weight of the edge (default: 1.0).
  /// @param allow_missing Whether to create missing nodes (default: false).
  ///
  /// @return true if a new edge was inserted.
  bool add_edge(
    const std::string &src,
    const std::string &tgt,
    double weight = 1.0,
    bool allow_missing = false
  );

  /// @brief Set whether the network is directed.
  void set_directed(const bool directed);

  /// @brief Set whether the network is a multigraph.
  void set_multigraph(const bool multigraph);
  
  /// @brief Read an edge list from a file and build the network.
  ///
  /// @param file_name Path to the file.
  /// @param has_headers Whether the file has a header row (default: false).
  /// @param sep Field delimiter (default: tab).
  /// @param directed Whether the network should be directed (default: false).
  /// @param multigraph Whether to allow multiple edges per pair (default: false).
  void read_edge_list(
    const std::string &file_name,
    bool has_headers = false,
    char sep = '\t',
    bool directed = false,
    bool multigraph = false
  );

  /// @brief Return the sum of each column in the adjacency matrix.
  ///
  /// @param[out] col_sums Output vector of column sums.
  /// @param label_list Optional list of node labels to include (default: all).
  void get_col_sums(
    std::vector<double> &col_sums,
    const std::vector<std::string> &label_list = {}
  ) const;
  
  /// @brief Get the weight of an edge between two nodes.
  ///
  /// @param src Source node label.
  /// @param tgt Target node label.
  ///
  /// @return Edge weight, or NaN if edge not found.
  double get_edge_weight(
    const std::string &src,
    const std::string &tgt
  ) const;

  /// @brief Get the adjacency matrix (CSR) of the network.
  ///
  /// @param label_list Optional list of node labels to restrict the matrix (default: all nodes in network).
  ///
  /// @return CSR-formatted adjacency matrix.
  virtual CSR_Matrix get_adjacency_matrix(const std::vector<std::string> &label_list = {}) const;

  /// @brief Get the transition matrix (CSR) used in Markov processes.
  ///
  /// @param label_list Optional list of node labels to restrict the matrix.
  ///
  /// @return CSR-formatted transition matrix.
  ///
  /// @throws std::runtime_error if called on a directed network.
  CSR_Matrix get_transition_matrix(const std::vector<std::string> &label_list = {}) const;

  /// @brief Print the internal network structure to a file for debugging.
  /// @param file_name Path to the output file.
  void print(const std::string& file_name) const;
  
  static Network merge_networks(const std::vector<Network>& networks, MergeMethod method);

  void convert_edges_to_distance();

  std::vector<std::vector<std::string>>
  find_all_shortest_paths(
    const std::string& s,
    const std::unordered_set<std::string>& T = {},
    bool use_weights = true
  ) const;

  virtual std::vector<std::vector<std::string>>
  find_all_shortest_paths(
    const std::unordered_set<std::string>& S,
    const std::unordered_set<std::string>& T = {},
    bool use_weights = true
  ) const;

  std::vector<std::vector<std::string>> find_all_shortest_paths_bfs(
    const std::string& s,
    const std::unordered_set<std::string>& T = {}
  ) const;

  virtual std::vector<std::vector<std::string>> find_all_shortest_paths_bfs(
    const std::unordered_set<std::string>& S,
    const std::unordered_set<std::string>& T = {}
  ) const;

  std::vector<std::vector<std::string>> find_all_shortest_paths_dijkstra(
    const std::string& s,
    const std::unordered_set<std::string>& T = {}
  ) const;

  virtual std::vector<std::vector<std::string>> find_all_shortest_paths_dijkstra(
    const std::unordered_set<std::string>& S,
    const std::unordered_set<std::string>& T = {}
  ) const;

  std::vector<std::vector<std::string>> connected_components() const;

  double get_total_edge_weight() const { return directed_ ? total_edge_weight_ : total_edge_weight_ / 2.0; };

protected:
  /// @brief Add an edge between two node indices with a given weight.
  ///
  /// @param src_idx Index of the source node.
  /// @param tgt_idx Index of the target node.
  /// @param weight  Weight of the edge.
  ///
  /// @throws std::out_of_range if src_idx or tgt_idx is invalid.
  virtual void add_edge_core(
    uint32_t src_idx,
    uint32_t tgt_idx,
    double weight
  );

  /// @brief Update the weight of an existing edge if the new weight is greater.
  ///
  /// @param src_idx Index of the source node.
  /// @param tgt_idx Index of the target node.
  /// @param weight  New weight to possibly assign.
  ///
  /// @return true if the edge was updated, false otherwise.
  ///
  /// @throws std::out_of_range if src_idx or tgt_idx is invalid.
  virtual bool update_edge(
    uint32_t src_idx,
    uint32_t tgt_idx,
    double weight
  );

  /// @brief Project a global label list into the network's local node set.
  ///
  /// @param label_list List of labels to validate or restrict (default: all nodes in network).
  ///
  /// @return A vector of valid labels for this network.
  std::vector<std::string> create_local_label_list(const std::vector<std::string> &label_list = {}) const;
  
  /// @brief Determine the size (rows, columns, nnz) of the transition matrix.
  ///
  /// @param[out] n_rows Number of rows.
  /// @param[out] n_cols Number of columns.
  /// @param[out] nnz Number of non-zero entries.
  /// @param label_list Optional list of node labels to restrict matrix size (default: all nodes in network).
  void get_transition_matrix_size(
    std::size_t &n_rows,
    std::size_t &n_cols,
    std::size_t &nnz,
    const std::vector<std::string> &label_list = {}
  ) const;

  static MergeMethod merge_method_from_string(const std::string& s);
  static std::string merge_method_to_string(MergeMethod m);
  static uint64_t pack_edge(uint32_t src, uint32_t tgt);

  void reconstruct_paths(
    uint32_t source_idx,
    const std::unordered_set<uint32_t>& nodes_to_backtrack,
    const std::vector<std::vector<uint32_t>>& predecessors,
    std::vector<std::vector<uint32_t>>& result
  ) const;

  virtual std::vector<std::vector<uint32_t>> find_all_shortest_paths_bfs_core(
    uint32_t s_idx,
    const std::unordered_set<uint32_t>& target_indices = {}
  ) const;

  virtual std::vector<std::vector<uint32_t>> find_all_shortest_paths_dijkstra_core(
    uint32_t s_idx,
    const std::unordered_set<uint32_t>& target_indices = {}
  ) const;

  virtual std::vector<std::vector<std::string>> convert_paths_to_labels(const std::vector<std::vector<uint32_t>>& index_paths) const;

private:
  bool multigraph_ = false;
  bool directed_ = false;
  std::size_t nnz_ = 0;
  std::size_t max_degree_ = 0;

  double total_edge_weight_ = 0.0;

  std::unordered_map<std::string, uint32_t> labels_; // lookup: label -> index
  std::vector<std::string> index_to_label_; // reverse lookup: index -> label
  std::vector<std::vector<Edge>> edges_;
};
