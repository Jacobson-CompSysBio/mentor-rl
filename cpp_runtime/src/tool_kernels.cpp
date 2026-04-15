#include "tool_kernels.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace mentor {

namespace {

std::string json_escape(const std::string& value) {
  std::ostringstream stream;
  for (const char ch : value) {
    switch (ch) {
      case '\\':
        stream << "\\\\";
        break;
      case '"':
        stream << "\\\"";
        break;
      case '\n':
        stream << "\\n";
        break;
      case '\r':
        stream << "\\r";
        break;
      case '\t':
        stream << "\\t";
        break;
      default:
        stream << ch;
        break;
    }
  }
  return stream.str();
}

std::string json_quote(const std::string& value) {
  return "\"" + json_escape(value) + "\"";
}

std::string json_bool(const bool value) {
  return value ? "true" : "false";
}

std::string make_ok_result(
    const std::string& payload_json,
    const std::string& provenance_json,
    const bool is_empty) {
  std::ostringstream stream;
  stream << "{\"ok\":true,\"payload\":" << payload_json << ",\"provenance\":" << provenance_json
         << ",\"is_empty\":" << json_bool(is_empty) << "}";
  return stream.str();
}

std::string make_error_result(const std::string& message) {
  return "{\"ok\":false,\"error\":" + json_quote(message) + "}";
}

template <typename ValueType>
std::vector<ValueType> unique_preserving_order(const std::vector<ValueType>& values) {
  std::vector<ValueType> unique_values;
  std::unordered_set<ValueType> seen;
  for (const auto& value : values) {
    if (seen.insert(value).second) {
      unique_values.push_back(value);
    }
  }
  return unique_values;
}

std::vector<std::size_t> resolve_layer_indices(
    const GraphStore& store,
    const std::vector<std::string>& layer_names) {
  if (layer_names.empty()) {
    std::vector<std::size_t> indices(store.num_layers());
    std::iota(indices.begin(), indices.end(), 0U);
    return indices;
  }

  std::vector<std::size_t> indices;
  for (const auto& layer_name : unique_preserving_order(layer_names)) {
    indices.push_back(store.layer_index(layer_name));
  }
  return indices;
}

std::vector<std::uint32_t> resolve_present_gene_indices(
    const GraphStore& store,
    const std::vector<std::string>& gene_ids) {
  std::vector<std::uint32_t> indices;
  for (const auto& gene_id : unique_preserving_order(gene_ids)) {
    if (store.has_gene(gene_id)) {
      indices.push_back(store.gene_index(gene_id));
    }
  }
  return indices;
}

std::vector<std::string> stringify_gene_indices(
    const GraphStore& store,
    const std::vector<std::uint32_t>& indices) {
  std::vector<std::string> gene_ids;
  gene_ids.reserve(indices.size());
  for (const auto index : indices) {
    gene_ids.push_back(store.gene_id(index));
  }
  return gene_ids;
}

std::string json_string_array(const std::vector<std::string>& values) {
  std::ostringstream stream;
  stream << "[";
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index > 0) {
      stream << ",";
    }
    stream << json_quote(values[index]);
  }
  stream << "]";
  return stream.str();
}

std::string json_ranked_results(
    const GraphStore& store,
    const std::vector<std::pair<std::uint32_t, double>>& ranked_pairs) {
  std::ostringstream stream;
  stream << "[";
  for (std::size_t index = 0; index < ranked_pairs.size(); ++index) {
    if (index > 0) {
      stream << ",";
    }
    stream << "{\"gene_id\":" << json_quote(store.gene_id(ranked_pairs[index].first))
           << ",\"score\":" << ranked_pairs[index].second << "}";
  }
  stream << "]";
  return stream.str();
}

std::vector<double> row_weight_sums(const LayerCsr& layer) {
  const auto num_nodes = layer.indptr.empty() ? 0U : layer.indptr.size() - 1U;
  std::vector<double> sums(num_nodes, 0.0);
  for (std::size_t row = 0; row < num_nodes; ++row) {
    const auto start = static_cast<std::size_t>(layer.indptr[row]);
    const auto end = static_cast<std::size_t>(layer.indptr[row + 1]);
    double total = 0.0;
    for (auto offset = start; offset < end; ++offset) {
      total += static_cast<double>(layer.weights[offset]);
    }
    sums[row] = total;
  }
  return sums;
}

std::vector<double> personalized_pagerank(
    const LayerCsr& layer,
    const std::vector<std::uint32_t>& seed_indices,
    const double restart_probability,
    const std::size_t max_iterations = 50U,
    const double tolerance = 1e-8) {
  const auto num_nodes = layer.indptr.empty() ? 0U : layer.indptr.size() - 1U;
  std::vector<double> scores(num_nodes, 0.0);
  if (seed_indices.empty()) {
    return scores;
  }

  const auto seed_mass = 1.0 / static_cast<double>(seed_indices.size());
  std::vector<double> personalization(num_nodes, 0.0);
  for (const auto seed_index : seed_indices) {
    personalization[seed_index] = seed_mass;
  }

  scores = personalization;
  const auto degree = row_weight_sums(layer);

  for (std::size_t iteration = 0; iteration < max_iterations; ++iteration) {
    std::vector<double> next_scores(num_nodes, 0.0);
    for (std::size_t node = 0; node < num_nodes; ++node) {
      next_scores[node] = restart_probability * personalization[node];
    }

    for (std::size_t row = 0; row < num_nodes; ++row) {
      if (scores[row] == 0.0 || degree[row] == 0.0) {
        continue;
      }
      const auto start = static_cast<std::size_t>(layer.indptr[row]);
      const auto end = static_cast<std::size_t>(layer.indptr[row + 1]);
      const double base_share = (1.0 - restart_probability) * scores[row] / degree[row];
      for (auto offset = start; offset < end; ++offset) {
        const auto target = layer.indices[offset];
        next_scores[target] += base_share * static_cast<double>(layer.weights[offset]);
      }
    }

    double delta = 0.0;
    for (std::size_t index = 0; index < num_nodes; ++index) {
      delta += std::abs(next_scores[index] - scores[index]);
    }
    scores.swap(next_scores);
    if (delta < tolerance) {
      break;
    }
  }

  return scores;
}

std::vector<std::pair<std::uint32_t, double>> top_k_scores(
    const std::vector<double>& scores,
    const std::size_t top_k) {
  std::vector<std::pair<std::uint32_t, double>> ranked;
  ranked.reserve(scores.size());
  for (std::size_t index = 0; index < scores.size(); ++index) {
    ranked.push_back({static_cast<std::uint32_t>(index), scores[index]});
  }
  std::sort(
      ranked.begin(),
      ranked.end(),
      [](const auto& left, const auto& right) {
        if (left.second != right.second) {
          return left.second > right.second;
        }
        return left.first < right.first;
      });
  if (ranked.size() > top_k) {
    ranked.resize(top_k);
  }
  return ranked;
}

}  // namespace

std::string json_store_summary(const GraphStore& store) {
  std::vector<std::string> layer_names;
  layer_names.reserve(store.layers().size());
  for (const auto& layer : store.layers()) {
    layer_names.push_back(layer.name);
  }

  std::vector<std::string> gene_ids = store.gene_ids();
  std::ostringstream stream;
  stream << "{\"ok\":true,\"payload\":{\"num_genes\":" << store.num_genes()
         << ",\"num_layers\":" << store.num_layers()
         << ",\"layer_names\":" << json_string_array(layer_names)
         << ",\"gene_ids\":" << json_string_array(gene_ids)
         << "},\"provenance\":{\"backend\":\"cpp_raw_csr\"},\"is_empty\":false}";
  return stream.str();
}

std::string json_get_neighbors(
    const GraphStore& store,
    const std::string& gene_id,
    const std::vector<std::string>& layer_names) {
  try {
    const auto gene_index = store.gene_index(gene_id);
    const auto selected_layers = resolve_layer_indices(store, layer_names);

    std::unordered_set<std::uint32_t> neighbor_union_set;
    std::vector<std::string> queried_layer_names;
    std::ostringstream layers_stream;
    layers_stream << "[";
    for (std::size_t layer_offset = 0; layer_offset < selected_layers.size(); ++layer_offset) {
      const auto& layer = store.layers().at(selected_layers[layer_offset]);
      queried_layer_names.push_back(layer.name);

      std::vector<std::string> neighbors;
      const auto start = static_cast<std::size_t>(layer.indptr[gene_index]);
      const auto end = static_cast<std::size_t>(layer.indptr[gene_index + 1]);
      neighbors.reserve(end - start);
      for (auto offset = start; offset < end; ++offset) {
        const auto neighbor_index = layer.indices[offset];
        neighbor_union_set.insert(neighbor_index);
        neighbors.push_back(store.gene_id(neighbor_index));
      }

      if (layer_offset > 0) {
        layers_stream << ",";
      }
      layers_stream << "{\"layer_name\":" << json_quote(layer.name)
                    << ",\"neighbors\":" << json_string_array(neighbors)
                    << ",\"neighbor_count\":" << neighbors.size() << "}";
    }
    layers_stream << "]";

    std::vector<std::uint32_t> unique_neighbor_indices(neighbor_union_set.begin(), neighbor_union_set.end());
    std::sort(unique_neighbor_indices.begin(), unique_neighbor_indices.end());
    const auto unique_neighbors = stringify_gene_indices(store, unique_neighbor_indices);

    std::ostringstream payload;
    payload << "{\"query_gene_id\":" << json_quote(gene_id)
            << ",\"layers\":" << layers_stream.str()
            << ",\"unique_neighbors\":" << json_string_array(unique_neighbors)
            << ",\"unique_neighbor_count\":" << unique_neighbors.size() << "}";

    std::ostringstream provenance;
    provenance << "{\"tool_name\":\"get_neighbors\",\"queried_layers\":"
               << json_string_array(queried_layer_names) << "}";
    return make_ok_result(payload.str(), provenance.str(), unique_neighbors.empty());
  } catch (const std::exception& error) {
    return make_error_result(error.what());
  }
}

std::string json_induce_subgraph(
    const GraphStore& store,
    const std::vector<std::string>& gene_ids,
    const std::vector<std::string>& layer_names) {
  try {
    const auto selected_layers = resolve_layer_indices(store, layer_names);
    const auto query_gene_ids = unique_preserving_order(gene_ids);

    std::vector<std::string> present_gene_ids;
    std::vector<std::string> missing_gene_ids;
    std::unordered_set<std::uint32_t> selected_set;
    std::vector<std::uint32_t> selected_indices;

    for (const auto& gene_id : query_gene_ids) {
      if (store.has_gene(gene_id)) {
        const auto index = store.gene_index(gene_id);
        selected_set.insert(index);
        selected_indices.push_back(index);
        present_gene_ids.push_back(gene_id);
      } else {
        missing_gene_ids.push_back(gene_id);
      }
    }

    std::vector<std::string> queried_layer_names;
    std::ostringstream layers_stream;
    std::size_t combined_edge_count = 0U;
    layers_stream << "[";
    for (std::size_t layer_offset = 0; layer_offset < selected_layers.size(); ++layer_offset) {
      const auto& layer = store.layers().at(selected_layers[layer_offset]);
      queried_layer_names.push_back(layer.name);

      std::vector<std::string> layer_present_gene_ids;
      std::ostringstream edges_stream;
      std::size_t edge_count = 0U;
      edges_stream << "[";

      for (const auto source_index : selected_indices) {
        const auto row_start = static_cast<std::size_t>(layer.indptr[source_index]);
        const auto row_end = static_cast<std::size_t>(layer.indptr[source_index + 1]);
        if (row_start != row_end) {
          layer_present_gene_ids.push_back(store.gene_id(source_index));
        }
        for (auto offset = row_start; offset < row_end; ++offset) {
          const auto target_index = layer.indices[offset];
          if (selected_set.find(target_index) == selected_set.end()) {
            continue;
          }
          if (source_index >= target_index) {
            continue;
          }
          if (edge_count > 0) {
            edges_stream << ",";
          }
          edges_stream << "{\"source_gene_id\":" << json_quote(store.gene_id(source_index))
                       << ",\"target_gene_id\":" << json_quote(store.gene_id(target_index))
                       << ",\"weight\":" << static_cast<double>(layer.weights[offset]) << "}";
          ++edge_count;
        }
      }

      edges_stream << "]";
      std::sort(layer_present_gene_ids.begin(), layer_present_gene_ids.end());
      layer_present_gene_ids.erase(
          std::unique(layer_present_gene_ids.begin(), layer_present_gene_ids.end()),
          layer_present_gene_ids.end());

      if (layer_offset > 0) {
        layers_stream << ",";
      }
      layers_stream << "{\"layer_name\":" << json_quote(layer.name)
                    << ",\"present_gene_ids\":" << json_string_array(layer_present_gene_ids)
                    << ",\"edges\":" << edges_stream.str()
                    << ",\"edge_count\":" << edge_count << "}";
      combined_edge_count += edge_count;
    }
    layers_stream << "]";

    std::ostringstream payload;
    payload << "{\"query_gene_ids\":" << json_string_array(query_gene_ids)
            << ",\"present_gene_ids\":" << json_string_array(present_gene_ids)
            << ",\"missing_gene_ids\":" << json_string_array(missing_gene_ids)
            << ",\"layers\":" << layers_stream.str()
            << ",\"combined_edge_count\":" << combined_edge_count << "}";

    std::ostringstream provenance;
    provenance << "{\"tool_name\":\"induce_subgraph\",\"queried_layers\":"
               << json_string_array(queried_layer_names) << "}";
    return make_ok_result(payload.str(), provenance.str(), combined_edge_count == 0U);
  } catch (const std::exception& error) {
    return make_error_result(error.what());
  }
}

std::string json_shortest_path(
    const GraphStore& store,
    const std::string& source_gene_id,
    const std::string& target_gene_id,
    const std::string& layer_name) {
  try {
    const LayerCsr* layer = &store.aggregate_layer();
    std::vector<std::string> queried_layers;
    std::string search_mode = "aggregate_multiplex";

    if (!layer_name.empty()) {
      layer = &store.layers().at(store.layer_index(layer_name));
      queried_layers.push_back(layer_name);
      search_mode = "single_layer";
    } else {
      for (const auto& layer_item : store.layers()) {
        queried_layers.push_back(layer_item.name);
      }
    }

    const auto source_index = store.gene_index(source_gene_id);
    const auto target_index = store.gene_index(target_gene_id);
    const auto num_nodes = layer->indptr.size() - 1U;

    std::vector<int> parent(num_nodes, -1);
    std::vector<bool> visited(num_nodes, false);
    std::queue<std::uint32_t> queue;
    queue.push(source_index);
    visited[source_index] = true;

    while (!queue.empty() && !visited[target_index]) {
      const auto node = queue.front();
      queue.pop();
      const auto start = static_cast<std::size_t>(layer->indptr[node]);
      const auto end = static_cast<std::size_t>(layer->indptr[node + 1]);
      for (auto offset = start; offset < end; ++offset) {
        const auto neighbor = layer->indices[offset];
        if (!visited[neighbor]) {
          visited[neighbor] = true;
          parent[neighbor] = static_cast<int>(node);
          queue.push(neighbor);
        }
      }
    }

    std::vector<std::string> path_gene_ids;
    if (visited[target_index]) {
      for (int current = static_cast<int>(target_index); current != -1; current = parent[current]) {
        path_gene_ids.push_back(store.gene_id(static_cast<std::uint32_t>(current)));
      }
      std::reverse(path_gene_ids.begin(), path_gene_ids.end());
    }

    std::ostringstream payload;
    payload << "{\"source_gene_id\":" << json_quote(source_gene_id)
            << ",\"target_gene_id\":" << json_quote(target_gene_id)
            << ",\"path_gene_ids\":" << json_string_array(path_gene_ids)
            << ",\"hop_count\":";
    if (path_gene_ids.empty()) {
      payload << "null";
    } else {
      payload << (path_gene_ids.size() - 1U);
    }
    payload << ",\"layer_name\":";
    if (layer_name.empty()) {
      payload << "null";
    } else {
      payload << json_quote(layer_name);
    }
    payload << "}";

    std::ostringstream provenance;
    provenance << "{\"tool_name\":\"shortest_path\",\"search_mode\":"
               << json_quote(search_mode)
               << ",\"queried_layers\":" << json_string_array(queried_layers)
               << ",\"distance_type\":\"unweighted_hops\"}";
    return make_ok_result(payload.str(), provenance.str(), path_gene_ids.empty());
  } catch (const std::exception& error) {
    return make_error_result(error.what());
  }
}

std::string json_rwr_monoplex(
    const GraphStore& store,
    const std::vector<std::string>& seed_gene_ids,
    const std::string& layer_name,
    const std::size_t top_k,
    const double restart_probability) {
  try {
    const auto& layer = store.layers().at(store.layer_index(layer_name));
    const auto unique_seed_ids = unique_preserving_order(seed_gene_ids);
    const auto present_seed_indices = resolve_present_gene_indices(store, unique_seed_ids);
    std::vector<std::uint32_t> active_seed_indices;
    active_seed_indices.reserve(present_seed_indices.size());
    for (const auto seed_index : present_seed_indices) {
      const auto row_start = static_cast<std::size_t>(layer.indptr[seed_index]);
      const auto row_end = static_cast<std::size_t>(layer.indptr[seed_index + 1]);
      if (row_start != row_end) {
        active_seed_indices.push_back(seed_index);
      }
    }
    const auto active_seed_gene_ids = stringify_gene_indices(store, active_seed_indices);
    const auto scores = personalized_pagerank(layer, active_seed_indices, restart_probability);
    const auto ranked = top_k_scores(scores, top_k);

    std::ostringstream payload;
    payload << "{\"seed_gene_ids\":" << json_string_array(unique_seed_ids)
            << ",\"active_seed_gene_ids\":" << json_string_array(active_seed_gene_ids)
            << ",\"layer_name\":" << json_quote(layer_name)
            << ",\"top_k\":" << top_k
            << ",\"results\":" << json_ranked_results(store, ranked) << "}";

    std::ostringstream provenance;
    provenance << "{\"tool_name\":\"rwr_monoplex\",\"layer_name\":" << json_quote(layer_name)
               << ",\"algorithm\":\"personalized_pagerank\",\"restart_probability\":"
               << restart_probability << "}";
    return make_ok_result(payload.str(), provenance.str(), ranked.empty());
  } catch (const std::exception& error) {
    return make_error_result(error.what());
  }
}

std::string json_rwr_multiplex(
    const GraphStore& store,
    const std::vector<std::string>& seed_gene_ids,
    const std::size_t top_k,
    const double restart_probability) {
  try {
    const auto unique_seed_ids = unique_preserving_order(seed_gene_ids);
    const auto num_nodes = store.num_genes();
    std::vector<double> aggregated_scores(num_nodes, 0.0);
    std::vector<std::string> active_layers;
    std::unordered_set<std::string> active_seed_gene_ids_set;

    for (const auto& layer : store.layers()) {
      const auto active_seed_indices = resolve_present_gene_indices(store, unique_seed_ids);
      std::vector<std::uint32_t> layer_seed_indices;
      layer_seed_indices.reserve(active_seed_indices.size());
      for (const auto seed_index : active_seed_indices) {
        const auto start = static_cast<std::size_t>(layer.indptr[seed_index]);
        const auto end = static_cast<std::size_t>(layer.indptr[seed_index + 1]);
        if (start != end) {
          layer_seed_indices.push_back(seed_index);
          active_seed_gene_ids_set.insert(store.gene_id(seed_index));
        }
      }
      if (layer_seed_indices.empty()) {
        continue;
      }

      active_layers.push_back(layer.name);
      const auto layer_scores = personalized_pagerank(layer, layer_seed_indices, restart_probability);
      for (std::size_t index = 0; index < layer_scores.size(); ++index) {
        aggregated_scores[index] += layer_scores[index];
      }
    }

    if (!active_layers.empty()) {
      for (auto& score : aggregated_scores) {
        score /= static_cast<double>(active_layers.size());
      }
    }

    const auto ranked = top_k_scores(aggregated_scores, top_k);
    std::vector<std::string> active_seed_gene_ids(
        active_seed_gene_ids_set.begin(), active_seed_gene_ids_set.end());
    std::sort(active_seed_gene_ids.begin(), active_seed_gene_ids.end());

    std::ostringstream payload;
    payload << "{\"seed_gene_ids\":" << json_string_array(unique_seed_ids)
            << ",\"active_seed_gene_ids\":" << json_string_array(active_seed_gene_ids)
            << ",\"active_layers\":" << json_string_array(active_layers)
            << ",\"top_k\":" << top_k
            << ",\"results\":" << json_ranked_results(store, ranked) << "}";

    std::ostringstream provenance;
    provenance << "{\"tool_name\":\"rwr_multiplex\",\"algorithm\":\"mean_personalized_pagerank\""
               << ",\"restart_probability\":" << restart_probability
               << ",\"active_layers\":" << json_string_array(active_layers)
               << ",\"layer_count\":" << active_layers.size() << "}";
    return make_ok_result(payload.str(), provenance.str(), ranked.empty());
  } catch (const std::exception& error) {
    return make_error_result(error.what());
  }
}

}  // namespace mentor
