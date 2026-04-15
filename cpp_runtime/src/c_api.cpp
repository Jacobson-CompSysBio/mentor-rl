#include "graph_store.hpp"
#include "tool_kernels.hpp"

#include <cstring>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::vector<std::string> split_csv(const char* raw_value) {
  std::vector<std::string> parts;
  if (raw_value == nullptr) {
    return parts;
  }

  std::stringstream stream(raw_value);
  std::string part;
  while (std::getline(stream, part, ',')) {
    if (!part.empty()) {
      parts.push_back(part);
    }
  }
  return parts;
}

char* copy_string(const std::string& value) {
  auto* buffer = new char[value.size() + 1U];
  std::memcpy(buffer, value.c_str(), value.size() + 1U);
  return buffer;
}

std::string wrap_open_error(const std::string& message) {
  return "{\"ok\":false,\"error\":\"" + message + "\"}";
}

}  // namespace

extern "C" {

void* mentor_open_store(const char* store_dir) {
  try {
    auto store = std::make_unique<mentor::GraphStore>(mentor::GraphStore::load(store_dir));
    return store.release();
  } catch (...) {
    return nullptr;
  }
}

void mentor_close_store(void* handle) {
  auto* store = static_cast<mentor::GraphStore*>(handle);
  delete store;
}

char* mentor_store_summary(void* handle) {
  if (handle == nullptr) {
    return copy_string(wrap_open_error("Store handle is null."));
  }
  auto* store = static_cast<mentor::GraphStore*>(handle);
  return copy_string(mentor::json_store_summary(*store));
}

char* mentor_get_neighbors(void* handle, const char* gene_id, const char* layers_csv) {
  if (handle == nullptr) {
    return copy_string(wrap_open_error("Store handle is null."));
  }
  auto* store = static_cast<mentor::GraphStore*>(handle);
  return copy_string(mentor::json_get_neighbors(*store, gene_id == nullptr ? "" : gene_id, split_csv(layers_csv)));
}

char* mentor_induce_subgraph(void* handle, const char* genes_csv, const char* layers_csv) {
  if (handle == nullptr) {
    return copy_string(wrap_open_error("Store handle is null."));
  }
  auto* store = static_cast<mentor::GraphStore*>(handle);
  return copy_string(mentor::json_induce_subgraph(*store, split_csv(genes_csv), split_csv(layers_csv)));
}

char* mentor_shortest_path(
    void* handle,
    const char* source_gene_id,
    const char* target_gene_id,
    const char* layer_name) {
  if (handle == nullptr) {
    return copy_string(wrap_open_error("Store handle is null."));
  }
  auto* store = static_cast<mentor::GraphStore*>(handle);
  return copy_string(
      mentor::json_shortest_path(
          *store,
          source_gene_id == nullptr ? "" : source_gene_id,
          target_gene_id == nullptr ? "" : target_gene_id,
          layer_name == nullptr ? "" : layer_name));
}

char* mentor_rwr_monoplex(
    void* handle,
    const char* seeds_csv,
    const char* layer_name,
    unsigned long long top_k,
    double restart_probability) {
  if (handle == nullptr) {
    return copy_string(wrap_open_error("Store handle is null."));
  }
  auto* store = static_cast<mentor::GraphStore*>(handle);
  return copy_string(
      mentor::json_rwr_monoplex(
          *store,
          split_csv(seeds_csv),
          layer_name == nullptr ? "" : layer_name,
          static_cast<std::size_t>(top_k),
          restart_probability));
}

char* mentor_rwr_multiplex(
    void* handle,
    const char* seeds_csv,
    unsigned long long top_k,
    double restart_probability) {
  if (handle == nullptr) {
    return copy_string(wrap_open_error("Store handle is null."));
  }
  auto* store = static_cast<mentor::GraphStore*>(handle);
  return copy_string(
      mentor::json_rwr_multiplex(
          *store,
          split_csv(seeds_csv),
          static_cast<std::size_t>(top_k),
          restart_probability));
}

void mentor_free_string(char* value) {
  delete[] value;
}

}  // extern "C"
