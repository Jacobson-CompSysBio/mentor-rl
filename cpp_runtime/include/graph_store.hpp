#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace mentor {

struct LayerCsr {
  std::string name;
  std::vector<std::uint64_t> indptr;
  std::vector<std::uint32_t> indices;
  std::vector<float> weights;
  std::uint32_t node_count = 0;
  std::uint64_t undirected_edge_count = 0;
  std::uint64_t stored_nnz = 0;
};

class GraphStore {
 public:
  static GraphStore load(const std::string& store_dir);

  const std::vector<std::string>& gene_ids() const { return gene_ids_; }
  const std::vector<LayerCsr>& layers() const { return layers_; }
  const LayerCsr& aggregate_layer() const { return aggregate_layer_; }
  std::uint32_t num_genes() const { return static_cast<std::uint32_t>(gene_ids_.size()); }
  std::uint32_t num_layers() const { return static_cast<std::uint32_t>(layers_.size()); }

  bool has_gene(const std::string& gene_id) const;
  bool has_layer(const std::string& layer_name) const;
  std::uint32_t gene_index(const std::string& gene_id) const;
  std::size_t layer_index(const std::string& layer_name) const;
  const std::string& gene_id(std::uint32_t index) const;

 private:
  std::vector<std::string> gene_ids_;
  std::unordered_map<std::string, std::uint32_t> gene_to_index_;
  std::vector<LayerCsr> layers_;
  std::unordered_map<std::string, std::size_t> layer_name_to_index_;
  LayerCsr aggregate_layer_;
};

}  // namespace mentor
