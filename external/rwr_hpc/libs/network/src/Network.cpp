#include "network/Network.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <set>
#include <queue>
#include <cmath>
#include <exception>
#include <assert.h>
#include <queue>
#include <stack>


#ifdef USE_OPENMP
#include <omp.h>
#else
inline int omp_get_max_threads() { return 1; }
inline int omp_get_thread_num() { return 0; }
#endif

#include <sort/sort.hpp>

// std::vector<std::string> Network::get_labels() const {
//   std::vector<std::string> output;
//   output.reserve(labels_.size());

//   for (const auto& [label, _] : labels_) {
//     output.push_back(label);
//   }
//   std::sort(output.begin(), output.end());

//   return output;
// }

uint32_t Network::get_idx_of_label(const std::string &label) const {
  auto search = labels_.find(label);
  if (search == labels_.end()) {
    throw std::runtime_error("Network::get_idx_of_label - could not find label in labels_");
  }
  return search->second;
}

std::size_t Network::get_nnz(const std::vector<std::string>& node_label_list) const {
  // If node_label_list is empty, assume all nnz values should be counted
  if (node_label_list.empty()) {
    return nnz_;
  }

  // Map labels in input list to their indicies
  std::unordered_set<std::size_t> included_indices;
  for (const auto& label : node_label_list) {
    auto it = labels_.find(label);
    if (it != labels_.end()) {
      included_indices.insert(it->second);
    }
  }

  // If all nodes are included, return total nnz
  if (included_indices.size() == labels_.size()) {
    return nnz_;
  }

  // Count edges with both src and tgt in input list
  std::size_t tmp_nnz = 0;
  for (std::size_t src = 0; src < edges_.size(); ++src) {
    if (!included_indices.count(src)) continue;

    for (const auto& edge : edges_[src]) {
      if (included_indices.count(edge.target)) {
        ++tmp_nnz;
      }
    }
  }

  return tmp_nnz;
}

std::size_t Network::degree(std::size_t idx) const {
  if (idx >= edges_.size()) {
    throw std::out_of_range("Network::degree - trying to access vertex index outside of range");
  }

  if (!directed_) {
    return edges_[idx].size();
  } else {
    throw std::runtime_error("Network::degree - not implemented for directed networks");
  }
}

std::size_t Network::add_node(const std::string &label) {
  // Get iterator to inserted element and bool inidacting if insertion occured 
  const auto [it, success] = labels_.insert(std::make_pair(label, labels_.size()));

  // If insertion occured add new element to edges vector
  if (success) {
    edges_.emplace_back();
    index_to_label_.push_back(label);
  }

  return it->second;
}

void Network::add_nodes(const std::vector<std::string> &labels) {
  for (auto &label : labels) {
    add_node(label);
  }
}

bool Network::add_edge(const std::string &src, const std::string &tgt, double weight, bool allow_missing) {
  // Do not add an edge if the weight is 0.0
  if (weight == 0.0) return false;

  // Get index of 'src' node and 'tgt' node. If 'allow_missing' is
  // false and either is missing throw
  uint32_t src_idx, tgt_idx;
  try {
    src_idx = get_idx_of_label(src);
  } catch (...) {
    if (allow_missing) src_idx = add_node(src);
    else throw;
  }
  try {
    tgt_idx = get_idx_of_label(tgt);
  } catch (...) {
    if (allow_missing) tgt_idx = add_node(tgt);
    else throw;
  }
  
  bool inserted = false;

  // If network is a multigraph add edge
  if (multigraph_) {
    add_edge_core(src_idx, tgt_idx, weight);
    if (!directed_) add_edge_core(tgt_idx, src_idx, weight);
    inserted = true;
  } else {
    // Update edge if it exists
    bool edge_updated = update_edge(src_idx, tgt_idx, weight);

    // Add edge if it does not exist
    if (!edge_updated) {
      add_edge_core(src_idx, tgt_idx, weight);
      inserted = true;
    }

    if (!directed_) {
      // Update edge if it exists
      bool edge_updated = update_edge(tgt_idx, src_idx, weight);

      // Add edge it it does not exist
      if (!edge_updated) {
        add_edge_core(tgt_idx, src_idx, weight);
        inserted = true;
      }
    }
  }

  if (!directed_) {
    max_degree_ = std::max({max_degree_, degree(src_idx), degree(tgt_idx)});
  }

  return inserted;
}

void Network::set_directed(bool directed) {
  directed_ = directed;
}

void Network::set_multigraph(bool multigraph) {
  multigraph_ = multigraph;
}

void Network::read_edge_list(
  const std::string& file_name,
  bool has_headers,
  char sep,
  bool directed,
  bool multigraph
) {
  // Try to open file
  std::ifstream input(file_name);
  if (!input) {
    throw std::runtime_error("Input file (" + file_name + ") could not be opened");
  }

  directed_ = directed;
  multigraph_ = multigraph;
  std::string line, src, tgt;
  
  // Read in headers if indicated
  if (has_headers && !std::getline(input, line)) {
    throw std::runtime_error("File missing expected header line");
  }

  labels_.clear();
  index_to_label_.clear();
  edges_.clear();
  uint32_t next_label = 0;

  // Lambda function. Adds 'label' with 'next_label' to unordered map if
  // 'label' is not already in map. If 'label' is added, the 'next_label'
  // is incremented. The value associated with 'label' is returned
  auto get_or_add = [&](const std::string& label) -> std::size_t {
    auto [it, inserted] = labels_.emplace(label, next_label);
    if (inserted) {
      index_to_label_.push_back(label);
      edges_.emplace_back();
      ++next_label;
    }
    return it->second;
  };

  while (std::getline(input, line)) {
    if (line.empty()) continue;

    std::istringstream iss(line);
    std::string weight_str;
    double weight = 1.0;

    std::getline(iss, src, sep);
    std::getline(iss, tgt, sep);

    // Check if row has third column
    if (iss.peek() != EOF) {
      std::getline(iss, weight_str, sep);

      if (!weight_str.empty()) {
        // Try to convert weight_str to double. throw if conversion fails
        try {
          weight = std::stod(weight_str);
        } catch (std::runtime_error&) {
          throw std::runtime_error("Invalid weight value un edge list: " + weight_str);
        }
      }
    }

    // Add the edge to local container. Store src and tgt in terms
    // of index associated with the labels
    std::size_t src_idx = get_or_add(src);
    std::size_t tgt_idx = get_or_add(tgt);

    if (src_idx == tgt_idx) {
      fprintf(stderr, "Self loop in layer %s\n", file_name.c_str()); 
      continue;
    }
   
    if (multigraph_) {
      add_edge_core(src_idx, tgt_idx, weight);
      if (!directed_) add_edge_core(tgt_idx, src_idx, weight);
    } else {
      bool updated = update_edge(src_idx, tgt_idx, weight);
      if (!updated) add_edge_core(src_idx, tgt_idx, weight);

      if (!directed_) {
        updated = update_edge(tgt_idx, src_idx, weight);
        if (!updated) add_edge_core(tgt_idx, src_idx, weight);
      }
    }

    if (!directed_) {
      max_degree_ = std::max({ max_degree_, degree(src_idx), degree(tgt_idx) });
    }
  }

  input.close();
}

void Network::get_col_sums(
  std::vector<double> &col_sums,
  const std::vector<std::string> &label_list
) const {
  auto local_label_list = create_local_label_list(label_list);
  std::size_t n_cols = local_label_list.size();

  // Initialize each column sum to zero
  col_sums.resize(n_cols);
  if (n_cols == 0) return;
  
  col_sums.assign(n_cols, 0.0);

  // Build fast lookup: label → local column index
  std::unordered_map<std::string, std::size_t> label_to_col;
  for (std::size_t i = 0; i < local_label_list.size(); ++i) {
    label_to_col[local_label_list[i]] = i;
  }

  // Prepare a per-thread buffer for partial sums
  const int n_threads = omp_get_max_threads();
  std::vector<std::vector<double>> thread_col_sums(n_threads, std::vector<double>(n_cols, 0.0));

  #pragma omp parallel
  {
    const int tid = omp_get_thread_num();
    auto& local_sums = thread_col_sums[tid];

    #pragma omp for schedule(dynamic)
    for (std::size_t src_idx = 0; src_idx < edges_.size(); ++src_idx){

      #ifndef NDEBUG
        if (src_idx >= index_to_label_.size()) {
          throw std::out_of_range("src_idx out of range");
        }
      #endif
      

      // Check if src_label in int label_list
      const std::string& src_label = index_to_label_[src_idx];
      if (label_to_col.find(src_label) == label_to_col.end()) continue;

      for (const Edge& e : edges_[src_idx]) {
        #ifndef NDEBUG
          if (e.target >= index_to_label_.size()) {
            throw std::out_of_range("tgt_idx out of range");
          }
        #endif

        const std::string& tgt_label = index_to_label_[e.target];

        auto it = label_to_col.find(tgt_label);
        if (it != label_to_col.end()) {
          local_sums[it->second] += e.weight;
        } 
      }
    } 
  }

  // Reduce thread-local results into col_sums
  for (const auto& local : thread_col_sums) {
    for (std::size_t j = 0; j < n_cols; ++j) {
      col_sums[j] += local[j];
    }
  }
}

double Network::get_edge_weight(
  const std::string &src,
  const std::string &tgt
) const {
  // Check if network contains src label
  // If not return weight as nan
  auto src_it = labels_.find(src);
  if (src_it == labels_.end()){
    return std::nan("");
  }

  // Check if network contains tgt label
  // If not return weight as nan
  auto tgt_it = labels_.find(tgt);
  if (tgt_it == labels_.end()) {
    return std::nan("");
  }
  
  // Loop through all edges incident to vertex 'src
  // If an edge was 'tgt' as the target return the weight
  for (auto &e : edges_[src_it->second]) {
    if (e.target == tgt_it->second) {
      return e.weight;
    }
  }

  // Return nan since no edge was found
  return std::nan("");
}

CSR_Matrix Network::get_adjacency_matrix(const std::vector<std::string> &label_list) const {
  // Get the size of the adjacency matrix
  std::size_t n_rows, n_cols, nnz;
  get_transition_matrix_size(n_rows, n_cols, nnz, label_list);
 
  // Create local_label_list 
  auto local_label_list = create_local_label_list(label_list);
  const std::size_t N = local_label_list.size();

  // Build label -> index lookup table for local_label_list
  std::unordered_map<std::string, std::size_t> label_to_local_idx;
  for (std::size_t i = 0; i < N; ++i) {
    label_to_local_idx[local_label_list[i]] = i;
  }

  // Initialize the adjaceny matrix
  CSR_Matrix adj_matrix(n_rows, n_cols, nnz);

  // Loop through all node labels in the local label list
  for (std::size_t i = 0; i < N; ++i) {
    const std::string& src_label = local_label_list[i];

    // Check if label is in 'labels_' (network)
    auto it = labels_.find(src_label);
    if (it == labels_.end()) continue;
    
    // Add all edges with targets in local_label_list to container
    std::size_t src_idx = it->second;
    std::vector<std::pair<std::size_t, double>> row_edges;
    for (const Edge& e : edges_[src_idx]) {
      // Check if the target node in in the local_label_list
      if (e.target >= index_to_label_.size()) continue; // e.target out of bounds

      const std::string& tgt_label = index_to_label_[e.target];
      auto tgt_it = label_to_local_idx.find(tgt_label);
      if (tgt_it != label_to_local_idx.end()) {
        row_edges.emplace_back(tgt_it->second, e.weight);
      }
    }

    // Check if at least one edge had a targt in the local_label_list
    if (!row_edges.empty()) {
      // Sort edeges by target index
      std::sort(row_edges.begin(), row_edges.end(),
                sort::SortPairByFirstItemIncreasing());

      // Add edges in row to matrix
      adj_matrix.add_row(i, row_edges);
    }
  }

  return adj_matrix;
}

CSR_Matrix Network::get_transition_matrix(const std::vector<std::string> &label_list) const {
  if (directed_) {
    throw std::runtime_error("Network::get_transition_matrix - transition matrix is not implemented for directed networks");
  } else {
    CSR_Matrix tran_matrix = get_adjacency_matrix(label_list);
    tran_matrix.col_normalize();
    return tran_matrix;
  }
}

void Network::print(const std::string& file_name) const {
  // Try to open file
  std::ofstream out(file_name);
  if (!out) {
    throw std::runtime_error("Network::print - Could not open file: " + file_name);
  }

  out << "labels_ (label → index):\n";
  for (const auto& [label, index] : labels_) {
    out << label << ": " << index << '\n';
  }

  out << "\nindex_to_label_ (index → label):\n";
  for (std::size_t i = 0; i < index_to_label_.size(); ++i) {
    out << i << ": " << index_to_label_[i] << '\n';
  }

  out << "\nedges_ (src_index → [target_index, weight]):\n";
  for (std::size_t src = 0; src < edges_.size(); ++src) {
    out << src << ": ";
    for (const auto& edge : edges_[src]) {
      out << "[" << edge.target << ", " << edge.weight << "] ";
    }
    out << '\n';
  }

  out.close();
}

void Network::add_edge_core(uint32_t src_idx, uint32_t tgt_idx, double weight) {
  if (src_idx >= edges_.size()) {
    throw std::out_of_range("Network::add_edge_core - src_idx is out of range");
  }
  if (tgt_idx >= edges_.size()) {
    throw std::out_of_range("Network::add_edge_core - tgt_idx is out of range");
  }

  edges_[src_idx].push_back(Edge{tgt_idx, weight});
  ++nnz_;
  total_edge_weight_ += weight;
}

bool Network::update_edge(uint32_t src_idx, uint32_t tgt_idx, double weight) {
  if (src_idx >= edges_.size()) {
    throw std::out_of_range("Network::update_edge - src_idx is out of range");
  }
  if (tgt_idx >= edges_.size()) {
    throw std::out_of_range("Network::update_edge - tgt_idx is out of range");
  }

  for (auto& [_target, _weight] : edges_[src_idx]) {
    if (_target == tgt_idx) {
      // Keep the largest weight
      if (weight > _weight){
        total_edge_weight_ -= _weight;
        total_edge_weight_ += weight;

        _weight = weight;
      } 
      return true;
    }
  }

  return false;
}

std::vector<std::string> Network::create_local_label_list(const std::vector<std::string> &label_list) const {
  return label_list.empty() ? get_labels() : label_list;
}

void Network::get_transition_matrix_size(
  std::size_t &n_rows,
  std::size_t &n_cols,
  std::size_t &nnz,
  const std::vector<std::string> &label_list
) const {
  // If label_list is empty assume all nodes are requested
  if (label_list.empty()) {
    // Calculate n_rows and n_cols directly
    n_rows = labels_.size();
    n_cols = n_rows;

    // Get nnz
    nnz = get_nnz();
  } else {
    // Calculate n_rows and n_cols directly
    n_rows = label_list.size();
    n_cols = n_rows;

    // Get nnz
    nnz = get_nnz(label_list);
  }
}

Network Network::merge_networks(const std::vector<Network>& networks, MergeMethod method) {
  Network merged_net;
  if (method == MergeMethod::All) {
    merged_net.set_multigraph(true);
  }

  // 1) Gather all labels and determine unique labels across all layers
  std::set<std::string> all_labels;
  for (const auto& net: networks) {
    all_labels.insert(net.index_to_label_.begin(), net.index_to_label_.end());
  }
  std::size_t N = all_labels.size();

  // 2) Create index_to_label_ and labels_ 
  merged_net.index_to_label_.reserve(N);
  merged_net.labels_.reserve(N);

  std::size_t i = 0;
  for (const auto& n : all_labels) {
    merged_net.index_to_label_.push_back(n);
    merged_net.labels_[n] = i++;
  }

  merged_net.edges_.resize(N); // Pre-allocate edges

  // 3) Build remap per layer
  std::vector<std::vector<uint32_t>> remap(networks.size());
  #ifdef USE_OPENMP
  #pragma omp parallel for schedule(dynamic)
  #endif
  for (std::size_t i = 0; i < networks.size(); ++i) {
    const auto& net = networks[i];
    auto& r = remap[i];
    r.resize(net.index_to_label_.size());
    for (uint32_t old = 0; old < r.size(); ++old) {
      r[old] = merged_net.labels_[net.index_to_label_[old]];
    }
  }

  // 4) Handle "all" method
  if (method == MergeMethod::All) {
    #ifdef USE_OPENMP
    #pragma omp parallel
    #endif
    {
      std::vector<std::vector<Edge>> local(N);
      #ifdef USE_OPENMP
      #pragma omp for schedule(dynamic)
      #endif
      for (std::size_t i = 0; i < networks.size(); ++i) {
        const auto& net = networks[i];
        const auto& r   = remap[i];
        for (uint32_t src_old = 0; src_old < net.edges_.size(); ++src_old) {
          uint32_t src_new = r[src_old];
          for (const Edge& e : net.edges_[src_old]) {
            uint32_t tgt_new = r[e.target];
            local[src_new].push_back({tgt_new, e.weight});
          }
        }
      }
      
      #ifdef USE_OPENMP
      #pragma omp critical
      #endif
      for (uint32_t s = 0; s < N; ++s) {
        auto& dst = merged_net.edges_[s];
        auto& src = local[s];
        dst.insert(dst.end(), src.begin(), src.end());

        merged_net.nnz_ += src.size();
      }
    }

    return merged_net;  
  }

  // 5) Thread-local aggregation for max/min/sum/mean methods
  int T = omp_get_max_threads();  
  std::vector<std::unordered_map<uint64_t, double>> weight_local(T);
  std::vector<std::unordered_map<uint64_t, uint32_t>> count_local(method == MergeMethod::Mean ? T : 1);

  #ifdef USE_OPENMP
  #pragma omp parallel
  #endif
  {
    int tid = omp_get_thread_num();
    auto& WL = weight_local[tid];
    auto& CL = (method == MergeMethod::Mean ? count_local[tid] : count_local[0]);
    
    WL.reserve(N);
    if (method == MergeMethod::Mean) {
      CL.reserve(N);
    }

    #ifdef USE_OPENMP
    #pragma omp for schedule(dynamic)
    #endif
    for (std::size_t i = 0; i < networks.size(); ++i) {
      const auto& net = networks[i];

      double total_layer_edge_weight = net.get_total_edge_weight();

      const auto& r   = remap[i];
      for (uint32_t src_old = 0; src_old < net.edges_.size(); ++src_old) {
        uint32_t src_new = r[src_old];
        for (const Edge& e : net.edges_[src_old]) {
          uint32_t tgt_new = r[e.target];

          if (src_new == tgt_new) {
            fprintf(stderr, "Self loop with node %s in layer %lu\n", merged_net.index_to_label_[src_new].c_str(), i);
            continue;
          } 

          uint64_t key = pack_edge(src_new, tgt_new);

          double edge_weight = e.weight / total_layer_edge_weight;
          auto it = WL.find(key);
          if (it == WL.end()) {
            WL[key] = edge_weight;
            if (method == MergeMethod::Mean) {
              CL[key] = 1;
            }
          } else {
            switch (method) {
              case MergeMethod::Max:  it->second = std::max(it->second, edge_weight); break;
              case MergeMethod::Min:  it->second = std::min(it->second, edge_weight); break;
              case MergeMethod::Sum:  it->second += edge_weight; break;
              case MergeMethod::Mean: it->second += edge_weight; CL[key] += 1; break;
              default: break;
            }
          }
        }
      }
    }
  }

  // 6) Reduce thread-local maps
  std::unordered_map<uint64_t, double> agg;
  agg.reserve(2 * 500000 * T);

  std::unordered_map<uint64_t, uint32_t> count;
  if (method == MergeMethod::Mean) {
    count.reserve(2 * 500000 * T);
  }

  for (int t = 0; t < T; ++t) {
    for (auto& kv : weight_local[t]) {
      uint64_t key = kv.first;
      double w = kv.second;

      auto it = agg.find(key);
      if (it == agg.end()) {
        agg[key] = w;
        if (method == MergeMethod::Mean) {
          count[key] = count_local[t][key];
        }
      } else {
        switch (method) {
          case MergeMethod::Max: it->second = std::max(it->second, w); break;
          case MergeMethod::Min: it->second = std::min(it->second, w); break;
          case MergeMethod::Sum: it->second += w; break;
          case MergeMethod::Mean:
            it->second += w;
            count[key] += count_local[t][key];
            break;
          default: break;
        }
      }
    }
  }

  if (method == MergeMethod::Mean) {
    for (auto& kv : agg) {
      kv.second /= double(count[kv.first]);
    }
  }

  // 7) Convert map to adjacency
  for (auto& kv : agg) {
    uint64_t key = kv.first;
    uint32_t src = key >> 32;
    uint32_t tgt = uint32_t(key & 0xffffffffu);
    merged_net.edges_[src].push_back({tgt, kv.second});

    merged_net.nnz_ += 1;
  }

  return merged_net;
}

MergeMethod Network::merge_method_from_string(const std::string& s) {
  if (s == "max")  return MergeMethod::Max;
  if (s == "min")  return MergeMethod::Min;
  if (s == "all")  return MergeMethod::All;
  if (s == "sum")  return MergeMethod::Sum;
  if (s == "mean") return MergeMethod::Mean;

  throw std::invalid_argument("Network::merge_method_from_string - Invalid merge method: " + s +
                              ". Expected: max, min, all, sum, mean.");
}

std::string Network::merge_method_to_string(MergeMethod m) {
  switch (m) {
    case MergeMethod::Max:  return "max";
    case MergeMethod::Min:  return "min";
    case MergeMethod::All:  return "all";
    case MergeMethod::Sum:  return "sum";
    case MergeMethod::Mean: return "mean";
  }
  return "unknown"; // unreachable but avoids warnings
}

uint64_t Network::pack_edge(uint32_t src, uint32_t tgt) {
  return (uint64_t(src) << 32 | uint64_t(tgt));
}

void Network::convert_edges_to_distance() {
  for (auto& edge_list: edges_) {
    for (auto& edge : edge_list) {
      if (edge.weight > 1.0) {
        throw std::runtime_error("Network::convert_edges_to_distance - Cannot calculate distance for edge weight > 1.0");
      } else {
        edge.weight = 1.01 - edge.weight;
      }
    }
  }
}

void Network::reconstruct_paths(
  uint32_t source_idx,
  const std::unordered_set<uint32_t>& nodes_to_backtrack,
  const std::vector<std::vector<uint32_t>>& predecessors,
  std::vector<std::vector<uint32_t>>& result
) const {
  // Early exit if predecessors is empty
  if (predecessors.empty() || nodes_to_backtrack.empty()) return;

  struct StackFrame {
    uint32_t node;
    std::size_t pred_idx;       // index of the next predecessor to visit
    std::vector<uint32_t> path; // current path
  };
  
  for (uint32_t target : nodes_to_backtrack) {
    if (predecessors[target].empty()){
      fprintf(stderr, "oh oh...node has no predecessors!!!\n");
      continue;
    } 

    std::stack<StackFrame> stack;
    stack.push({target, 0, {target}});

    while (!stack.empty()) {
      auto& frame = stack.top();

      if (frame.path.size() > edges_.size()) {
        fprintf(stderr, "Cycle detected involving node %u\n", frame.node);
        std::abort();
      }


      if (frame.node == source_idx) {
        std::vector<uint32_t> rev_path = frame.path;
        std::reverse(rev_path.begin(), rev_path.end());
        result.push_back(std::move(rev_path));
        stack.pop();
        continue;
      }

      if (frame.pred_idx < predecessors[frame.node].size()) {
        uint32_t pred = predecessors[frame.node][frame.pred_idx++];
        std::vector<uint32_t> new_path = frame.path;
        new_path.push_back(pred);
        stack.push({pred, 0, std::move(new_path)});
      } else {
        stack.pop(); // all predecessors processed
      }
    }
  }
}

std::vector<std::vector<uint32_t>> Network::find_all_shortest_paths_bfs_core(
  uint32_t s_idx,
  const std::unordered_set<uint32_t>& target_indices
) const {
  const size_t n = edges_.size();
  assert(s_idx < n);

  // Distance from source to each node; UINT32_MAX means not visited
  std::vector<uint32_t> distance(n, UINT32_MAX);
  // Predecessors for reconstructing all shortest paths
  std::vector<std::vector<uint32_t>> predecessors(n);
  std::queue<uint32_t> q;

  // Make a local copy of targets and remove the source if present
  std::unordered_set<uint32_t> target_indices_ = target_indices;
  target_indices_.erase(s_idx);

  std::unordered_set<uint32_t> reached_targets;

  distance[s_idx] = 0;
  q.push(s_idx);

  int target_distance = -1;

  while (!q.empty()) {
    uint32_t u = q.front(); q.pop();
    uint32_t dist = distance[u];

    if (target_distance != -1 && dist > target_distance) {
      break;
    }

    for (const auto& edge : edges_[u]) {
      uint32_t v = edge.target;

      if (distance[v] == UINT32_MAX) {
        distance[v] = dist + 1;
        predecessors[v].push_back(u);
        q.push(v);
      } else if (distance[v] == dist + 1) {
        predecessors[v].push_back(u);
      }
    }

    // Update reached targets
    if (!target_indices_.empty() && target_indices_.count(u)) {
      reached_targets.insert(u);

      // Set target_distance once all targets are reached
      if (reached_targets.size() == target_indices_.size()) {
        target_distance = dist;
      }
    }
  }

  std::unordered_set<uint32_t> nodes_to_backtrack;
  if (target_indices_.empty()) {
    // If no targets specified, backtrack all reachable nodes except the source
    for (uint32_t i = 0; i < n; ++i) {
      if (i != s_idx && distance[i] != UINT32_MAX) {
        nodes_to_backtrack.insert(i);
      }
    }
  } else {
    nodes_to_backtrack = target_indices_;
  }
  
  std::vector<std::vector<uint32_t>> paths;
  reconstruct_paths(s_idx, nodes_to_backtrack, predecessors, paths);

  return paths;
}

std::vector<std::vector<uint32_t>> Network::find_all_shortest_paths_dijkstra_core(
  uint32_t s_idx,
  const std::unordered_set<uint32_t>& target_indices
) const {
  const size_t n = edges_.size();
  assert(s_idx < n);

  constexpr double INF = std::numeric_limits<double>::infinity();
  constexpr double EPS = 1e-9;

  std::vector<double> distance(n, INF);
  std::vector<std::vector<uint32_t>> predecessors(n);
  std::vector<std::vector<uint32_t>> paths;

  using PQNode = std::pair<double, uint32_t>;
  std::priority_queue<PQNode, std::vector<PQNode>, std::greater<PQNode>> pq;

  // Local copy we can safely modify
  std::unordered_set<uint32_t> targets = target_indices;
  targets.erase(s_idx);

  std::unordered_set<uint32_t> reached_targets;
  double target_distance = -1.0;

  // Store confirmed shortest-path distances for each reached target
  std::unordered_map<uint32_t,double> settled_target_dist;

  // Largest confirmed shortest-path distance among *settled* targets
  double max_target_dist = -1.0;

  distance[s_idx] = 0.0;
  pq.push({0.0, s_idx});

  while (!pq.empty()) {
    auto [dist_u, u] = pq.top(); pq.pop();

    // stale entry
    if (dist_u - EPS > distance[u]) continue;

    // If u is a target, its distance is now settled
    if (targets.count(u)) {
      settled_target_dist[u] = dist_u;

      // If we have settled all targets
      if (settled_target_dist.size() == targets.size()) {

        // Compute the maximum shortest-path distance of all targets
        max_target_dist = 0.0;
        for (auto &kv : settled_target_dist)
          max_target_dist = std::max(max_target_dist, kv.second);
      }
    }

    // Early-stop condition (Correct for multi-target Dijkstra):
    // Once all targets are settled, PQ.top() > max_target_dist ⇒ no shorter path remains
    if (max_target_dist >= 0.0 && std::abs(dist_u - max_target_dist) < EPS) {
      break;
    }


    // Relax edges
    for (const auto &edge : edges_[u]) {
      uint32_t v = edge.target;
      double new_dist = dist_u + edge.weight;

      // Relaxation cases
      // if (new_dist + EPS < distance[v]) {
      if (new_dist < distance[v]) {
        distance[v] = new_dist;
        predecessors[v].clear();
        predecessors[v].push_back(u);
        pq.push({new_dist, v});
      }
      else if (std::abs(new_dist - distance[v]) < EPS) {
        // Another predecessor with equal shortest distance
        predecessors[v].push_back(u);
      }
    }
  }

  // Determine which nodes to backtrack
  std::unordered_set<uint32_t> nodes_to_backtrack;
  if (targets.empty()) {
    for (uint32_t i = 0; i < n; ++i) {
      if (i != s_idx && distance[i] < std::numeric_limits<double>::infinity()) {
        nodes_to_backtrack.insert(i);
      }
    }
  } else {
    nodes_to_backtrack = targets;
  }

  reconstruct_paths(s_idx, nodes_to_backtrack, predecessors, paths);
  return paths;
}

std::vector<std::vector<std::string>> Network::convert_paths_to_labels(
  const std::vector<std::vector<uint32_t>>& index_paths
) const {
  std::vector<std::vector<std::string>> label_paths;
  label_paths.reserve(index_paths.size()); // Reserve space for outer vector

  for (auto& path : index_paths) {
    std::vector<std::string> lp;
    lp.reserve(path.size()); // Reserve space for inner vector

    for (auto idx : path){
      lp.push_back(index_to_label_[idx]);
    }

    label_paths.push_back(std::move(lp));
  }
  return label_paths;
}

std::vector<std::vector<std::string>> Network::find_all_shortest_paths(
  const std::string& s,
  const std::unordered_set<std::string>& T,
  bool use_weights
) const {
  return find_all_shortest_paths(std::unordered_set<std::string>{s}, T, use_weights);
}

std::vector<std::vector<std::string>> Network::find_all_shortest_paths(
  const std::unordered_set<std::string>& S,
  const std::unordered_set<std::string>& T,
  bool use_weights
) const {
  // If S is empty return empty vector
  if (S.empty()) {
    return {};
  }

  // Error checking (same behavior as BFS/Dijkstra public versions)
  for (const auto& s : S) {
    if (!labels_.count(s)) {
      throw std::invalid_argument(
        "Network::find_all_shortest_paths - Source node label '" + s + "' not found in the network.");
    }
  }
  for (const auto& t : T) {
    if (!labels_.count(t)) {
      throw std::invalid_argument(
        "Network::find_all_shortest_paths - Target node label '" + t + "' not found in the network.");
    }
  }

  // Branching logic
  if (!use_weights) {
    return find_all_shortest_paths_bfs(S, T);
  }

  // Use weights → check whether all weights are equal
  bool first = true;
  double w0  = 1.0;
  bool all_equal = true;

  for (const auto& edge_list : edges_) {
    for (const auto& e : edge_list) {
      if (first) {
        w0 = e.weight;
        first = false;
      } else if (std::abs(e.weight - w0) > 1e-12) {
        all_equal = false;
        break;
      }
    }
    if (!all_equal) break;
  }

  if (all_equal) {
    // Equivalent to unweighted graph BFS
    return find_all_shortest_paths_bfs(S, T);
  }

  return find_all_shortest_paths_dijkstra(S, T);
}


std::vector<std::vector<std::string>> Network::find_all_shortest_paths_bfs(
  const std::string& s,
  const std::unordered_set<std::string>& T
) const {
  // Check source label
  auto s_it = labels_.find(s);
  if (s_it == labels_.end()) {
    throw std::invalid_argument("Network::find_all_shortest_paths_bfs - Source node label '" + s + "' not found in the network.");
  }
  uint32_t s_idx = s_it->second;

  // Check target labels
  std::unordered_set<uint32_t> target_indices;
  for (const auto& t : T) {
    auto t_it = labels_.find(t);
    if (t_it == labels_.end()) {
      throw std::invalid_argument("Network::find_all_shortest_paths_bfs - Target node label '" + t + "' not found in the network.");
    }
    target_indices.insert(t_it->second);
  }

  auto index_paths = find_all_shortest_paths_bfs_core(s_idx, target_indices);
  return convert_paths_to_labels(index_paths);
}

std::vector<std::vector<std::string>> Network::find_all_shortest_paths_bfs(
  const std::unordered_set<std::string>& S,
  const std::unordered_set<std::string>& T
) const {
  // Precompute target indices with descriptive exceptions
  std::unordered_set<uint32_t> target_indices;
  for (const auto& t : T) {
    auto t_it = labels_.find(t);
    if (t_it == labels_.end()) {
      throw std::invalid_argument("Network::find_all_shortest_paths_bfs - Target node label '" + t + "' not found in the network.");
    }
    target_indices.insert(t_it->second);
  }

  // Convert sources to a vector for indexing in the parallel loop
  std::vector<std::string> sources(S.begin(), S.end());

  // Pre-check all sources before starting parallel computation
  std::vector<uint32_t> source_indices;
  source_indices.reserve(sources.size());
  for (const auto& s : sources) {
    auto s_it = labels_.find(s);
    if (s_it == labels_.end()) {
      throw std::invalid_argument("Network::find_all_shortest_paths_bfs - Source node label '" + s + "' not found in the network.");
    }
    source_indices.push_back(s_it->second);
  }

  std::vector<std::vector<std::string>> all_paths;

  #ifdef USE_OPENMP
  #pragma omp parallel
  #endif
  {
    std::vector<std::vector<std::string>> local_paths;

    #ifdef USE_OPENMP
    #pragma omp for nowait
    #endif
    for (size_t i = 0; i < source_indices.size(); ++i) {
      uint32_t s_idx = source_indices[i];

      auto index_paths = find_all_shortest_paths_bfs_core(s_idx, target_indices);
      auto label_paths = convert_paths_to_labels(index_paths);

      local_paths.insert(local_paths.end(), label_paths.begin(), label_paths.end());
    }

    #ifdef USE_OPENMP
    #pragma omp critical
    #endif
    all_paths.insert(all_paths.end(), local_paths.begin(), local_paths.end());
  }

  return all_paths;
}

std::vector<std::vector<std::string>> Network::find_all_shortest_paths_dijkstra(
  const std::string& s,
  const std::unordered_set<std::string>& T
) const {
  // Check source label
  auto s_it = labels_.find(s);
  if (s_it == labels_.end()) {
    throw std::invalid_argument("Network::find_all_shortest_paths_dijkstra - Source node label '" + s + "' not found in the network.");
  }
  uint32_t s_idx = s_it->second;

  // Check target labels
  std::unordered_set<uint32_t> target_indices;
  for (const auto& t : T) {
    auto t_it = labels_.find(t);
    if (t_it == labels_.end()) {
      throw std::invalid_argument("Network::find_all_shortest_paths_dijkstra - Target node label '" + t + "' not found in the network.");
    }
    target_indices.insert(t_it->second);
  }

  auto index_paths = find_all_shortest_paths_dijkstra_core(s_idx, target_indices);
  return convert_paths_to_labels(index_paths);
}

std::vector<std::vector<std::string>> Network::find_all_shortest_paths_dijkstra(
  const std::unordered_set<std::string>& S,
  const std::unordered_set<std::string>& T
) const {
  // Precompute target indices with descriptive exceptions
  std::unordered_set<uint32_t> target_indices;
  for (const auto& t : T) {
    auto t_it = labels_.find(t);
    if (t_it == labels_.end()) {
      throw std::invalid_argument("Network::find_all_shortest_paths_dijkstra - Target node label '" + t + "' not found in the network.");
    }
    target_indices.insert(t_it->second);
  }

  // Convert sources to a vector for indexing in the parallel loop
  std::vector<std::string> sources(S.begin(), S.end());

  // Pre-check all sources before starting parallel computation
  std::vector<uint32_t> source_indices;
  source_indices.reserve(sources.size());
  for (const auto& s : sources) {
    auto s_it = labels_.find(s);
    if (s_it == labels_.end()) {
      throw std::invalid_argument("Network::find_all_shortest_paths_dijkstra - Source node label '" + s + "' not found in the network.");
    }
    source_indices.push_back(s_it->second);
  }

  std::vector<std::vector<std::string>> all_paths;

  #ifdef USE_OPENMP
  #pragma omp parallel
  #endif
  {
    std::vector<std::vector<std::string>> local_paths;

    #ifdef USE_OPENMP
    #pragma omp for nowait
    #endif
    for (size_t i = 0; i < source_indices.size(); ++i) {
      uint32_t s_idx = source_indices[i];
      auto index_paths = find_all_shortest_paths_dijkstra_core(s_idx, target_indices);
      auto label_paths = convert_paths_to_labels(index_paths);

      local_paths.insert(local_paths.end(), label_paths.begin(), label_paths.end());
    }

    #ifdef USE_OPENMP
    #pragma omp critical
    #endif
    all_paths.insert(all_paths.end(), local_paths.begin(), local_paths.end());
  }

  return all_paths;
}


std::vector<std::vector<std::string>> Network::connected_components() const
{
  const std::size_t n = index_to_label_.size();
  std::vector<std::vector<std::string>> components;
  std::vector<bool> visited(n, false);
  components.reserve(n);

  for (std::size_t start = 0; start < n; ++start) {
    if (visited[start]) continue;

    // Start BFS for this connected component
    std::queue<std::size_t> q;
    q.push(start);
    visited[start] = true;

    std::vector<std::string> component;
    component.reserve(16);
    component.push_back(index_to_label_[start]);

    while (!q.empty()) {
      std::size_t u = q.front();
      q.pop();

      for (const Edge& e : edges_[u]) {
        std::size_t v = e.target;
        if (!visited[v]) {
          visited[v] = true;
          q.push(v);
          component.push_back(index_to_label_[v]);
        }
      }
    }

    components.push_back(std::move(component));
  }

  return components;
}
