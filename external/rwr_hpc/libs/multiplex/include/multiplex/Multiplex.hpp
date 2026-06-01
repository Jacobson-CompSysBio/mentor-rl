/// @file Multiplex.hpp
/// @brief Defines the Multiplex class for handling multi-layer network structures.
///
/// The Multiplex class provides an abstraction over multiple network layers, each represented
/// as a Network object. It supports construction from disk, selective layer/node filtering,
/// and CSR-format transition matrix generation.
/// 
/// @author Ken Smith
/// @date 2025-07-24

#pragma once

#include <vector>
#include <string>
#include <set>

#include <network/Network.hpp>
#include <sparse/CSR_Matrix.hpp>

/// @class Multiplex
/// @brief Represents a multiplex (multi-layer) network, where each layer is a separate Network object.
/// 
/// This class supports reading multiplex networks from file, tracking node-layer associations,
/// and building transition matrices used in random walk-based algorithms.
/// 
/// Layers can be selectively included, and node labels can be filtered as needed.
/// All matrices are returned in CSR format and follow consistent node-label ordering.
class Multiplex {
public:
  /// @brief Default constructor.
  Multiplex() = default;

  /// @brief Construct a multiplex network by reading layer definitions from file.
  ///
  /// @param file_name Path to the multiplex definition file.
  /// @param has_headers Whether the file includes a header row (default: false).
  explicit Multiplex(const std::string &file_name, bool has_headers = false);

  /// @brief Destructor
  ~Multiplex() = default;

  /// @brief Get the number of unique nodes across all layers.
  ///
  /// @return The total number of nodes.
  inline std::size_t n_nodes() const { return nodes_.size(); }

  /// @brief Get the number of layers in the multiplex.
  ///
  /// @return The number of Network layers.
  inline std::size_t n_layers() const { return networks_.size(); }

  /// @brief Get the list of all node labels in the multiplex.
  ///
  /// @return A vector of node label strings.
  std::vector<std::string> get_nodes() const;

  /// @brief Get the name of each layer.
  /// @return A vector of layer name strings.
  inline std::vector<std::string> get_layer_names() const { return layer_name_; }

  /// @brief Check presence of node labels in each layer.
  ///
  /// @param nodes_labels A list of node labels to check (default: all in multiplex).
  ///
  /// @return A boolean vector (n_layers × node_labels.size()) in column-major order indicating presence of each node in each layer.
  ///
  /// @note Rows represent layers and columns represent nodes.
  std::vector<bool> get_nodes_by_layer(const std::vector<std::string>& nodes_labels = {}) const;

  /// @brief Construct a CSR-format intra-layer transition matrix for the multiplex.
  ///
  /// @param delta Probability of jumping to the same node in a different layer (default: 0.5).
  /// @param label_list Subset of node labels to include (default: all).
  /// @param layer_list Subset of layers to include (default: all).
  ///
  /// @return A CSR_Matrix representing intra-layer transition probabilities.
  ///
  /// @throws std::invalid_argument if delta is not valid probabilty (between 0 and 1, inclusive)
  CSR_Matrix get_intra_layer_transition_matrix(const double delta = 0.5,
                                               const std::vector<std::string> &label_list = {},
                                               const std::vector<bool> &layer_list = {}) const;

  /// @brief Construct an inter-layer transition matrix (flattened) from a given intra-layer matrix.
  ///
  /// @param intra_tran The intra-layer CSR_Matrix used as a base.
  /// @param delta Probability of jumping to the same node in a different layer (default: 0.5).
  /// @param label_list Optional list of node labels (default: all).
  /// @param layer_list Optional layer mask (default: all).
  ///
  /// @return A flattened vector representing the inter-layer transition matrix in column-major order.
  ///
  /// @throws std::invalid_argument if delta is not valid probabilty (between 0 and 1, inclusive)
  std::vector<double> get_inter_layer_transition_matrix(const CSR_Matrix& intra_tran,
                                                        const double delta = 0.5,
                                                        const std::vector<std::string> &label_list = {},
                                                        const std::vector<bool> &layer_list = {}) const;

  std::size_t get_n_intra_edges() const;

  Network merge_layers(MergeMethod method) const;

  std::vector<std::pair<std::string, double>> get_layers_with_highest_weight(const std::string& src, const std::string& tgt) const;

  std::vector<bool> get_layer_contribution_per_component(const std::vector<std::vector<std::string>>& components) const;

  const Network& get_network(std::size_t network_index) const {
    if (network_index >= networks_.size()) {
      throw std::out_of_range("network_index is out of range");
    }

    return networks_[network_index];
  }
  
protected:
  /// @brief Read and parse a multiplex file list from disk.
  ///
  /// @param file_name Path to the file listing layers.
  /// @param has_headers Whether the file includes header rows.
  ///
  /// @throws std::invalid_argument if the file cannot be opened
  /// @throws std::runtime_error if the file is empty
  void read_flist(const std::string &file_name, bool has_headers = false);

  /// @brief Filter and project a global layer list into the local multiplex structure.
  ///
  /// @param layer_list A global layer selection mask.
  ///
  /// @return A filtered layer mask for the local layers.
  ///
  /// @throws std::invalid_argument if layer_list is not empty and size does not match
  ///         number of layers in the multiplex
  /// @note If layer_list is empty, mask includes all layers
  std::vector<bool> create_local_layer_list(const std::vector<bool> &layer_list) const;

  /// @brief Filter and project a global node label list into local labels.
  ///
  /// @param label_list A list of global node labels to project.
  ///
  /// @return A filtered list containing only local node labels.
  /// 
  /// @note If label_list is empty, all nodes in multiplex are returned
  std::vector<std::string> create_local_label_list(const std::vector<std::string> &label_list) const;

private:
  std::vector<Network> networks_;   ///< The list of layers (one Network per layer).
  std::vector<std::string> layer_name_;  ///< The name for each layer (aligned with networks_).
  std::set<std::string> nodes_; ///< Unique set of all node labels across layers.
};
