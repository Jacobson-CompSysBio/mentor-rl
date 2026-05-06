#include "graph_store.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace mentor {

namespace {

std::string join_path(const std::string& base_dir, const std::string& name) {
  if (base_dir.empty()) {
    return name;
  }
  if (base_dir.back() == '/') {
    return base_dir + name;
  }
  return base_dir + "/" + name;
}

template <typename T>
std::vector<T> read_binary_vector(const std::string& path) {
  std::ifstream handle(path, std::ios::binary | std::ios::ate);
  if (!handle) {
    throw std::runtime_error("Could not open binary file: " + path);
  }

  const auto end_position = handle.tellg();
  handle.seekg(0, std::ios::beg);
  const auto byte_size = static_cast<std::size_t>(end_position);
  if (byte_size % sizeof(T) != 0) {
    throw std::runtime_error("Binary file has invalid size for dtype: " + path);
  }

  std::vector<T> values(byte_size / sizeof(T));
  if (!values.empty()) {
    handle.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(byte_size));
  }
  return values;
}

std::vector<std::string> split_tab_line(const std::string& line) {
  std::vector<std::string> parts;
  std::stringstream stream(line);
  std::string part;
  while (std::getline(stream, part, '\t')) {
    parts.push_back(part);
  }
  return parts;
}

LayerCsr read_layer(
    const std::string& store_dir,
    const std::string& name,
    const std::string& indptr_file,
    const std::string& indices_file,
    const std::string& weights_file,
    std::uint32_t node_count,
    std::uint64_t undirected_edge_count,
    std::uint64_t stored_nnz) {
  LayerCsr layer;
  layer.name = name;
  layer.indptr = read_binary_vector<std::uint64_t>(join_path(store_dir, indptr_file));
  layer.indices = read_binary_vector<std::uint32_t>(join_path(store_dir, indices_file));
  layer.weights = read_binary_vector<float>(join_path(store_dir, weights_file));
  layer.node_count = node_count;
  layer.undirected_edge_count = undirected_edge_count;
  layer.stored_nnz = stored_nnz;

  if (layer.indices.size() != layer.weights.size()) {
    throw std::runtime_error("Layer indices and weights length mismatch for layer: " + name);
  }
  if (!layer.indptr.empty() && layer.indptr.back() != layer.indices.size()) {
    throw std::runtime_error("Layer indptr does not match nnz for layer: " + name);
  }

  const auto num_nodes = layer.indptr.empty() ? 0U : layer.indptr.size() - 1U;
  layer.degree_sums.resize(num_nodes, 0.0);
  for (std::size_t row = 0; row < num_nodes; ++row) {
    const auto start = static_cast<std::size_t>(layer.indptr[row]);
    const auto end = static_cast<std::size_t>(layer.indptr[row + 1]);
    double total = 0.0;
    for (auto offset = start; offset < end; ++offset) {
      total += static_cast<double>(layer.weights[offset]);
    }
    layer.degree_sums[row] = total;
  }
  return layer;
}

}  // namespace

GraphStore GraphStore::load(const std::string& store_dir) {
  GraphStore store;

  {
    std::ifstream gene_file(join_path(store_dir, "genes.tsv"));
    if (!gene_file) {
      throw std::runtime_error("Could not open genes.tsv in store directory.");
    }

    std::string line;
    while (std::getline(gene_file, line)) {
      if (line.empty()) {
        continue;
      }
      const auto parts = split_tab_line(line);
      if (parts.size() < 2) {
        throw std::runtime_error("Invalid line in genes.tsv: " + line);
      }
      const auto index = static_cast<std::uint32_t>(std::stoul(parts[0]));
      const auto& gene_id = parts[1];
      if (index != store.gene_ids_.size()) {
        throw std::runtime_error("genes.tsv must use consecutive zero-based indices.");
      }
      store.gene_to_index_[gene_id] = index;
      store.gene_ids_.push_back(gene_id);
    }
  }

  {
    std::ifstream layer_file(join_path(store_dir, "layers.tsv"));
    if (!layer_file) {
      throw std::runtime_error("Could not open layers.tsv in store directory.");
    }

    std::string line;
    bool header_skipped = false;
    while (std::getline(layer_file, line)) {
      if (line.empty()) {
        continue;
      }
      if (!header_skipped) {
        header_skipped = true;
        continue;
      }

      const auto parts = split_tab_line(line);
      if (parts.size() < 8) {
        throw std::runtime_error("Invalid line in layers.tsv: " + line);
      }

      const auto layer_index = static_cast<std::size_t>(std::stoul(parts[0]));
      const auto& layer_name = parts[1];
      const auto& indptr_file = parts[2];
      const auto& indices_file = parts[3];
      const auto& weights_file = parts[4];
      const auto node_count = static_cast<std::uint32_t>(std::stoul(parts[5]));
      const auto undirected_edge_count = static_cast<std::uint64_t>(std::stoull(parts[6]));
      const auto stored_nnz = static_cast<std::uint64_t>(std::stoull(parts[7]));

      if (layer_index != store.layers_.size()) {
        throw std::runtime_error("layers.tsv must use consecutive zero-based layer indices.");
      }

      auto layer = read_layer(
          store_dir,
          layer_name,
          indptr_file,
          indices_file,
          weights_file,
          node_count,
          undirected_edge_count,
          stored_nnz);
      if (!layer.indptr.empty() && layer.indptr.size() != store.gene_ids_.size() + 1U) {
        throw std::runtime_error("Layer indptr length does not match gene count for layer: " + layer_name);
      }

      store.layer_name_to_index_[layer_name] = layer_index;
      store.layers_.push_back(std::move(layer));
    }
  }

  store.aggregate_layer_ = read_layer(
      store_dir,
      "aggregate",
      "aggregate_indptr.bin",
      "aggregate_indices.bin",
      "aggregate_weights.bin",
      store.num_genes(),
      0,
      0);
  if (!store.aggregate_layer_.indptr.empty() &&
      store.aggregate_layer_.indptr.size() != store.gene_ids_.size() + 1U) {
    throw std::runtime_error("Aggregate indptr length does not match gene count.");
  }

  return store;
}

bool GraphStore::has_gene(const std::string& gene_id) const {
  return gene_to_index_.find(gene_id) != gene_to_index_.end();
}

bool GraphStore::has_layer(const std::string& layer_name) const {
  return layer_name_to_index_.find(layer_name) != layer_name_to_index_.end();
}

std::uint32_t GraphStore::gene_index(const std::string& gene_id) const {
  const auto it = gene_to_index_.find(gene_id);
  if (it == gene_to_index_.end()) {
    throw std::runtime_error("Unknown gene ID: " + gene_id);
  }
  return it->second;
}

std::size_t GraphStore::layer_index(const std::string& layer_name) const {
  const auto it = layer_name_to_index_.find(layer_name);
  if (it == layer_name_to_index_.end()) {
    throw std::runtime_error("Unknown layer name: " + layer_name);
  }
  return it->second;
}

const std::string& GraphStore::gene_id(std::uint32_t index) const {
  return gene_ids_.at(index);
}

}  // namespace mentor
