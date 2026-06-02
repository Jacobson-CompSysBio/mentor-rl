// add adapter header
#include "rwr_hpc_adapter.hpp"

#include <stdexcept>

namespace mentor {

RwrHpcRwrResult rwr_hpc_rwr_multiplex(
    const GraphStore& store,
    const std::vector<std::string>& seed_gene_ids,
    std::size_t top_k,
    double restart_probability) {
    (void)store;
    (void)seed_gene_ids;
    (void)top_k;
    (void)restart_probability;

    throw std::runtime_error("rwr_hpc_rwr_multiplex is not implemented yet")
    }

RwrHpcRwrResult rwr_hpc_rwr_monoplex(
    const GraphStore& store,
    const std::vector<std::string>& seed_gene_ids,
    const std::string& layer_name,
    std::size_t top_k,
    double restart_probability) {
  (void)store;
  (void)seed_gene_ids;
  (void)layer_name;
  (void)top_k;
  (void)restart_probability;

  throw std::runtime_error("rwr_hpc_rwr_monoplex is not implemented yet.");
}

RwrHpcShortestPathResult rwr_hpc_shortest_path(
    const GraphStore& store,
    const std::string& source_gene_id,
    const std::string& target_gene_id,
    const std::string& layer_name) {
  (void)store;
  (void)source_gene_id;
  (void)target_gene_id;
  (void)layer_name;

  throw std::runtime_error("rwr_hpc_shortest_path is not implemented yet.");
}

RwrHpcNeighborsResult rwr_hpc_get_neighbors(
    const GraphStore& store,
    const std::string& gene_id,
    const std::vector<std::string>& layer_names) {
  (void)store;
  (void)gene_id;
  (void)layer_names;

  throw std::runtime_error("rwr_hpc_get_neighbors is not implemented yet.");
}

}