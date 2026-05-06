#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "graph_store.hpp"

namespace mentor {

std::string json_store_summary(const GraphStore& store);

std::string json_get_neighbors(
    const GraphStore& store,
    const std::string& gene_id,
    const std::vector<std::string>& layer_names);

std::string json_induce_subgraph(
    const GraphStore& store,
    const std::vector<std::string>& gene_ids,
    const std::vector<std::string>& layer_names);

std::string json_shortest_path(
    const GraphStore& store,
    const std::string& source_gene_id,
    const std::string& target_gene_id,
    const std::string& layer_name);

std::string json_rwr_monoplex(
    const GraphStore& store,
    const std::vector<std::string>& seed_gene_ids,
    const std::string& layer_name,
    std::size_t top_k,
    double restart_probability);

std::string json_rwr_multiplex(
    const GraphStore& store,
    const std::vector<std::string>& seed_gene_ids,
    std::size_t top_k,
    double restart_probability);

}  // namespace mentor
