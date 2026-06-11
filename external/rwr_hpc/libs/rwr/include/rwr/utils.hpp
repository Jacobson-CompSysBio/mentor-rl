#pragma once

#include <vector>
#include <string>
#include <sparse/CSR_Matrix.hpp>

namespace rwr {
  void read_seeds(std::vector<std::vector<std::string>> &seeds, const std::string &file_name, const bool skip_first_col = true, const char delim = '\t');

  void write_seeds(const std::vector<std::vector<std::string>>& seeds,
                      const std::string& file_name,
                      const std::vector<std::string>& first_col = {},
                      const char delim = '\t');
  // std::vector<std::string> read_gene_set(const std::string& file_name, const char delim = '\t');
  void get_default_seed_set(std::vector<std::vector<std::string>>& seed_set, const std::vector<std::string>& node_labels);
  void get_grin_loo_seeds(std::vector<std::vector<std::string>>& seed_set,
                          std::vector<std::string>& loo_seeds,
                          const std::vector<std::string>& base_seeds,
                          const std::vector<std::string>& node_labels,
                          const std::size_t n_samples_in_null_dist = 100,
                          const uint64_t seed = 42);
  std::size_t get_encoding_size(const std::string& reduction_method, const std::size_t N, const std::size_t L);

  std::vector<double> get_init_probs(const std::vector<std::vector<std::string>> &seeds,
                                     const std::vector<std::string> &nodes_in_mp,
                                     const std::size_t L,
                                     const std::vector<double>& tau = {});
  // CSR_Matrix get_init_probs_sparse(const std::vector<std::vector<std::string>> &seeds_vec,
  //                                  const std::vector<std::string> &nodes_in_mp,
  //                                  const std::size_t L,
  //                                  const std::vector<double>& raw_tau= {});

  std::vector<std::string> flatten(const std::vector<std::vector<std::string>>& input);

  void remove_seeds_not_in_list(std::vector<std::vector<std::string>>& seeds, const std::vector<std::string>& list);
  void remove_duplicate_seed_vectors(std::vector<std::vector<std::string>>& seeds);

  void get_seed_split_for_all_workers(const int world_size,
                                      const std::size_t num_seeds,
                                      std::vector<unsigned long>& start_seed_vec,
                                      std::vector<unsigned long>& stop_seed_vec,
                                      std::vector<unsigned long>& num_encodings);

  void scale_node_values(CSR_Matrix& intra_tran, const std::size_t N, const std::size_t node_index, const double value);
}