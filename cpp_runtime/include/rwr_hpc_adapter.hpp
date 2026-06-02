//pragma once means only include this header file once in a single compilation unit, preventing multiple definitions and potential conflicts.
#pragma once

#include <string>
#include <cstdint>
#include <string>
#include <vector>

// include the graph store header, which contains the graph obj definitions
#include "graph_store.hpp"

// namespace is used to organize identifiers, prevent naming conflicts, etc.
namespace mentor {

// struct allows group of different typed variables under a single name
struct RwrHpcRankedGene {
    std::uint32_t gene_index;
    double score;
};

struct RwrHpcRwrResult {
    std::vector<std::string> seed_gene_ids;
    std::vector<std::string> active_seed_gene_ids;
    std::vector<std::string> active_layers;
    std::vector<RwrHpcRankedGene> ranked_genes;
    std::string algorithm;
};

struct RwrHpcShortestPathResult {
    std::string source_gene_id;
    std::string target_gene_id;
    std::vector<std::string> path_gene_ids;
    std::vector<std::string> queried_layers;
    std::string search_mode;
    std::string distance_type;
};

struct RwrHpcNeighborsResult {
    std::string query_gene_id;
    std::vector<std::string> queried_layers;
    std::vector<std::string> unique_neighbors;
};

RwrHpcRwrResult rwr_hpc_rwr_multiplex(
    const GraphStore& store,
    const std::vector<std::string>& seed_gene_ids,
    std::size_t top_k,
    double restart_probability);

RwrHpcRwrResult rwr_hpc_rwr_monoplex(
    const GraphStore& store,
    const std::vector<std::string>& seed_gene_ids,
    const std::string& layer_name,
    std::size_t top_k,
    double restart_probability);

RwrHpcShortestPathResult rwr_hpc_shortest_path(
    const GraphStore& store,
    const std::string& source_gene_id,
    const std::string& target_gene_id,
    const std::string& layer_name);

RwrHpcNeighborsResult rwr_hpc_get_neighbors(
    const GraphStore& store,
    const std::string& gene_id,
    const std::vector<std::string>& layer_names);
}