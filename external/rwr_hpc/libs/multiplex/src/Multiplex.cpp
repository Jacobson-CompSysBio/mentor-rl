#include "multiplex/Multiplex.hpp"

#ifdef USE_OPENMP
#include <omp.h>
#endif

#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>

Multiplex::Multiplex(const std::string &file_name, bool has_headers) {
  read_flist(file_name, has_headers);
}

void Multiplex::read_flist(const std::string &file_name, bool has_headers) {
  networks_.clear();
  layer_name_.clear();
  nodes_.clear();

  std::ifstream input(file_name);
  if (!input) {
    throw std::invalid_argument("Multiplex::read_flist - flist could not be opened");
  }

  std::vector<std::string> lines;
  std::string line;
  while (std::getline(input, line)) {
    if (!line.empty()) {
      lines.push_back(line);
    }
  }
  input.close();

  const std::size_t n_layers = lines.size();
  if (n_layers == 0) {
    throw std::runtime_error("Multiplex::read_flist - flist is empty");
  }

  // Validate column count
  for (std::size_t i = 0; i < n_layers; ++i) {
    std::size_t n_tabs = std::count(lines[i].begin(), lines[i].end(), '\t');
    if (n_tabs < 1) {
      throw std::runtime_error("Multiplex::read_flist - Line " + std::to_string(i) + " has fewer than 2 tab-separated columns");
    }
  }

  networks_.resize(n_layers);
  layer_name_.resize(n_layers);
  std::vector<std::unordered_set<std::string>> local_nodes(n_layers);
  for (std::size_t i = 0; i < n_layers; ++i) {
    local_nodes[i].reserve(50000);
  }

  #ifdef USE_OPENMP
  #pragma omp parallel for
  #endif
  for (std::size_t i = 0; i < n_layers; ++i) {
    std::istringstream iss(lines[i]);
    std::string path, name;
    std::getline(iss, path, '\t');
    std::getline(iss, name, '\t');

    layer_name_[i] = name;
    Network net;
    net.read_edge_list(path, has_headers, '\t', false, false);

    auto labels = net.get_labels();
    local_nodes[i].insert(labels.begin(), labels.end());

    networks_[i] = std::move(net);
  }

  for (const auto& labels : local_nodes) {
    nodes_.insert(labels.begin(), labels.end());
  }
}

std::vector<std::string> Multiplex::get_nodes() const {
  std::vector<std::string> v(nodes_.begin(), nodes_.end());
  return v;
}

std::vector<bool> Multiplex::get_nodes_by_layer(const std::vector<std::string>& nodes_labels) const {
  const auto local_label_list = create_local_label_list(nodes_labels);
  
  std::vector<bool> output(n_layers() * local_label_list.size());
  for (std::size_t i = 0; i < n_layers(); ++i) {
    for (std::size_t j = 0; j < local_label_list.size(); ++j) {
      output[i + j * n_layers()] = networks_[i].contains_node(local_label_list[j]);
    }
  }

  return output;
}

CSR_Matrix Multiplex::get_intra_layer_transition_matrix(const double delta,
                                                        const std::vector<std::string> &label_list,
                                                        const std::vector<bool> &layer_list) const {
  if (delta <= 0 || delta >= 1.0) {
    throw std::invalid_argument("Multiplex::get_intra_layer_transition_matrix - delta must be between 0.0 and 1.0");
  }

  // Get local constants
  const auto local_label_list = create_local_label_list(label_list);
  const auto local_layer_list = create_local_layer_list(layer_list);
  const std::size_t N = local_label_list.size();
  const std::size_t L = std::count(local_layer_list.begin(), local_layer_list.end(), true);
  const std::size_t NL = N * L;
  const double one_minus_delta = 1.0 - delta;

  std::vector<std::size_t> nnz_offset_per_layer(local_layer_list.size(), 0);
  std::vector<std::size_t> rows_offest_per_layer(local_layer_list.size(), 0);
  std::size_t intra_nnz = 0, total_rows = 0;
  for (std::size_t i_layer = 0; i_layer < local_layer_list.size(); ++i_layer) {
    nnz_offset_per_layer[i_layer] = intra_nnz;
    rows_offest_per_layer[i_layer] = total_rows;

    if (local_layer_list[i_layer]) {
      intra_nnz += networks_[i_layer].get_nnz(local_label_list);
      total_rows += N;
    }
  }

  CSR_Matrix intra_tran(NL, NL, intra_nnz);
  intra_tran.values_.resize(intra_nnz);
  intra_tran.col_idx_.resize(intra_nnz);
  
  // Parallel version
  #ifdef USE_OPENMP
  #pragma omp parallel for
  #endif
  for (std::size_t i_layer = 0; i_layer < local_layer_list.size(); ++i_layer) {
    if (!local_layer_list[i_layer]) continue;

    CSR_Matrix tmp = networks_[i_layer].get_transition_matrix(local_label_list);

    std::size_t nnz_offset = nnz_offset_per_layer[i_layer];
    std::size_t row_offset = rows_offest_per_layer[i_layer];
    std::size_t col_offset = N * std::count(local_layer_list.begin(), local_layer_list.begin() + i_layer, true);

    for (std::size_t k = 0; k < tmp.nnz(); ++k) {
      intra_tran.values_[nnz_offset + k] = tmp.values_[k] * one_minus_delta;
      intra_tran.col_idx_[nnz_offset + k] = tmp.col_idx_[k] + col_offset;
    }

    for (std::size_t r = 1; r < tmp.row_ptr_.size(); ++r) {
      intra_tran.row_ptr_[1 + row_offset + r - 1] = tmp.row_ptr_[r] + nnz_offset;
    }
  }

  // Set nnz and final row_ptr valie
  intra_tran.nnz_ = intra_tran.values_.size();

  return intra_tran;
}

std::vector<double> Multiplex::get_inter_layer_transition_matrix(const CSR_Matrix& intra_tran,
                                                                 const double delta,
                                                                 const std::vector<std::string> &label_list,
                                                                 const std::vector<bool> &layer_list) const {
  if (delta <= 0 || delta >= 1.0) {
    throw std::invalid_argument("Multiplex::get_inter_layer_transition_matrix - delta must be between 0.0 and 1.0");
  }

  // Get local constants
  const auto local_label_list = create_local_label_list(label_list);
  const auto local_layer_list = create_local_layer_list(layer_list);
  const std::size_t N = local_label_list.size();
  const std::size_t L = std::count(local_layer_list.begin(), local_layer_list.end(), true);

  std::vector<double> inter_tran;
  if (L > 1) {
    const double inter_layer_value = delta / (L - 1);
    const double hanging_node_value = 1.0 / (L - 1);

    // Initialize inter_train layers 
    inter_tran.resize(N*L, inter_layer_value);

    // Check for hanging nodes by looking for columns with no edge
    std::vector<bool> has_edge(N * L,false);
    for (auto &col : intra_tran.col_idx_) {
      has_edge[col] = true;
    }
    // Set any values corresponding to hanging nodes to 'hanging_node_value
    for (std::size_t j = 0; j < N*L; ++j) {
      if (!has_edge[j]) {
        inter_tran[j] = hanging_node_value;
      }
    }
  } else {
    inter_tran.resize(N*L, 0.0);
  }

  return inter_tran;
}

std::size_t Multiplex::get_n_intra_edges() const {
  std::size_t n_intra_edges = 0;

  for (auto& net : networks_) {
    n_intra_edges += net.get_n_edges();
  }
  
  return n_intra_edges;
}

std::vector<bool> Multiplex::create_local_layer_list(const std::vector<bool> &layer_list) const {
  std::vector<bool> local_layer_list;
  if (layer_list.empty()) {
    local_layer_list.resize(networks_.size(), true);
  } else if (layer_list.size() == networks_.size()) {
    local_layer_list = layer_list;
  } else {
    throw std::invalid_argument("Multiplex::create_local_layer_list - layer_list length does not match number of layers");
  }
  return local_layer_list;
}

std::vector<std::string> Multiplex::create_local_label_list(const std::vector<std::string> &label_list) const {
  std::vector<std::string> local_label_list;
  if (label_list.empty()) {
    local_label_list.reserve(nodes_.size());
    for (const auto& node : nodes_) {
      local_label_list.push_back(node);
    }
  } else {
    local_label_list = label_list;
  }
  return local_label_list;
}


Network Multiplex::merge_layers(MergeMethod method) const {
  return Network::merge_networks(networks_, method);
}

std::vector<std::pair<std::string, double>> Multiplex::get_layers_with_highest_weight(const std::string& src, const std::string& tgt) const {
  std::vector<std::pair<std::string, double>> best_layers;
  double best_weight = 0.0;

  for (std::size_t i = 0; i < networks_.size(); ++i) {
    auto weight = networks_[i].get_edge_weight(src, tgt);
    double total_layer_edge_weight = networks_[i].get_total_edge_weight();

    if (!std::isnan(weight)) {
      weight /= total_layer_edge_weight;

      if ( weight > best_weight) {
        best_weight = weight;
        best_layers.clear();
        best_layers.push_back(std::make_pair(layer_name_[i], weight));
      } else if (std::abs(weight - best_weight) < std::numeric_limits<double>::epsilon()) {
        best_layers.push_back(std::make_pair(layer_name_[i], weight));
      }
    }

  }
  return best_layers;
}

std::vector<bool> Multiplex::get_layer_contribution_per_component(const std::vector<std::vector<std::string>>& components) const {
  std::vector<bool> layer_contribution(n_layers() * components.size());

  for (std::size_t j = 0; j < components.size(); ++j) {
    const auto& comp = components[j];
    std::size_t offset = j * n_layers();

    for (std::size_t l = 0; l < n_layers(); ++l) {
      const auto& net = networks_[l];
      for (const auto& node : comp) {
        if (net.contains_node(node)) {
          layer_contribution[offset + l] = true;
          break;
        }
      }
    }
  }

  return layer_contribution;
}