#include "rwr/utils.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <string_view>
#include <cctype>
#include <numeric>
#include <iomanip> // Required for setprecision
#include <unordered_map>

#include <sampling/sampling.hpp>
#include <utils/vector_utils.hpp>
#include <split/split.hpp>


namespace {
  void add_loo(std::vector<std::string>& loo_vec,
               std::vector<std::vector<std::string>>& loo_set,
               const std::vector<std::string>& in_vec) {
  
    for (std::size_t i = 0; i < in_vec.size(); ++i) {
      loo_vec.push_back(in_vec[i]);

      std::vector<std::string> tmp;
      for (std::size_t j = 0; j < in_vec.size(); ++j) {
        if (i != j) {
          tmp.push_back(in_vec[j]);
        }
      }

      loo_set.push_back(std::move(tmp));
    }
  }
}

// Trim leading and trailing whitespace (from a string_view)
inline std::string_view trim(std::string_view sv) {
  const auto begin = sv.find_first_not_of(" \t\r\n");
  const auto end = sv.find_last_not_of(" \t\r\n");
  return (begin == std::string_view::npos) ? "" : sv.substr(begin, end - begin + 1);
}

// Robust CSV/TSV-style line splitting with quoting, trimming, escaped quotes, and validation
inline std::vector<std::string> split_line_robust(std::string_view line, char delim) {
  std::vector<std::string> tokens;
  std::string token;
  bool in_quotes = false;
  bool saw_quote = false;

  for (std::size_t i = 0; i < line.size(); ++i) {
    char c = line[i];
    if (c == '"') {
      if (in_quotes && i + 1 < line.size() && line[i + 1] == '"') {
        token += '"'; // escaped quote
        ++i;
      } else {
        in_quotes = !in_quotes;
        saw_quote = true;
      }
    } else if (c == delim && !in_quotes) {
      tokens.emplace_back(trim(token));
      token.clear();
      saw_quote = false;
    } else {
      token += c;
    }
  }

  if (in_quotes) {
    throw std::runtime_error("Malformed input: unmatched quote in line: " + std::string(line));
  }

  tokens.emplace_back(trim(token));
  return tokens;
}

void rwr::read_seeds(std::vector<std::vector<std::string>> &seeds,
                           const std::string &file_name,
                           const bool skip_first_col,
                           const char delim) {
  std::ifstream input(file_name);
  if (!input.is_open()) {
    throw std::runtime_error("rwr::read_seeds - could not open file: " + file_name);
  }

  seeds.clear();
  std::string line;
  size_t line_num = 0;

  while (std::getline(input, line)) {
    ++line_num;
    std::string_view sv = line;

    // Trim leading whitespace for comment detection
    sv = trim(sv);
    if (sv.empty() || sv.front() == '#') {
      continue;  // skip empty or comment lines
    }

    try {
      std::vector<std::string> tokens = split_line_robust(sv, delim);

      // Conditionally skip the first column
      if (skip_first_col) {
        if (tokens.size() < 2) {
          throw std::runtime_error("not enough columns to skip first");
        }
        tokens.erase(tokens.begin());
      }

      seeds.push_back(std::move(tokens));
    } catch (const std::exception &e) {
      throw std::runtime_error("rwr::read_seeds - Error on line " + std::to_string(line_num) + ": " + e.what());
    }
  }

  input.close();
}

void rwr::write_seeds(const std::vector<std::vector<std::string>>& seeds,
                      const std::string& file_name,
                      const std::vector<std::string>& first_col,
                      const char delim) {
  std::ofstream out(file_name);
  if (!out.is_open()) {
    throw std::runtime_error("rwr::write_seeds - could not open file: " + file_name);
  }

  if (!first_col.empty() && first_col.size() != seeds.size()) {
    throw std::runtime_error("rwr::write_seeds - first column is not empty and size does not match seed size()");
  }

  for (std::size_t i = 0; i < seeds.size(); ++i) {
    if (!first_col.empty()) {
      out << first_col[i] << delim;
    }

    for (std::size_t j = 0; j < seeds[i].size() - 1; ++j) {
      out << seeds[i][j] << delim;
    }
    out << seeds[i][seeds[i].size() - 1] << '\n';
  }

  out.close();
}

// std::vector<std::string> rwr::read_gene_set(const std::string& file_name, const char delim ) {
//   std::ifstream input(file_name);
//   if (!input.is_open()) {
//     throw std::runtime_error("RWR_Utils::read_gene_set - could not open file: " + file_name);
//   }

//   std::vector<std::string> output;

//   std::string line;
//   size_t line_num = 0;
  
//   while (std::getline(input, line)) {
//     ++line_num;
//     std::string_view sv = line;

//     // Trim leading whitespace for comment detection
//     sv = trim(sv);
//     if (sv.empty() || sv.front() == '#') {
//       continue;  // skip empty or comment lines
//     }

//     try {
//       std::vector<std::string> tokens = split_line_robust(sv, delim);
//       if (tokens.size() < 2) {
//         throw std::runtime_error("RWR_Utils::read_gene_set - line" + std::to_string(line_num) + " has less than two columns");
//       }
//       output.push_back(std::move(tokens[1]));
//     } catch (const std::exception &e) {
//       throw std::runtime_error("Error on line " + std::to_string(line_num) + ": " + e.what());
//     }
//   }

//   input.close();

//   return output;
// }

void rwr::get_default_seed_set(std::vector<std::vector<std::string>>& seed_set,
                               const std::vector<std::string>& node_labels) {
  seed_set.clear();
  for (auto &node : node_labels) {
    seed_set.push_back({node});
  }
}

void rwr::get_grin_loo_seeds(std::vector<std::vector<std::string>>& seed_set,
                             std::vector<std::string>& loo_seeds,
                             const std::vector<std::string>& base_seeds,
                             const std::vector<std::string>& node_labels,
                             const std::size_t n_samples_in_null_dist,
                             const uint64_t seed) {
  std::mt19937_64 rng(seed);

  add_loo(loo_seeds, seed_set, base_seeds);
  
  // Remove nodes in base_seeds
  auto node_labels_in_mp_copy = node_labels;
  utils::remove_elements(node_labels_in_mp_copy, loo_seeds);

  // Loop over the desired number of random samples for the null distribution
  std::vector<std::string> rnd_seeds;
  for (std::size_t i = 0; i < n_samples_in_null_dist; ++i) {
    // Sample without replacement
    sampling::sample_vector_without_replacement(rnd_seeds,
                                                node_labels_in_mp_copy,
                                                base_seeds.size(),
                                                rng);
    add_loo(loo_seeds, seed_set, rnd_seeds);
  }
}

std::size_t rwr::get_encoding_size(const std::string& reduction_method, const std::size_t N, const std::size_t L) {
  if (reduction_method.compare("geometric") == 0 ||
      reduction_method.compare("arithmetic") == 0 ||
      reduction_method.compare("sum") == 0) {
    return N;
  } else if (reduction_method.compare("none") == 0) {
    return N * L;
  } else {
    throw std::invalid_argument("rwr::get_encoding_size - unknown reduction method");
  }
}

std::vector<double> rwr::get_init_probs(const std::vector<std::vector<std::string>> &seeds_vec,
                                        const std::vector<std::string> &nodes_in_mp,
                                        const std::size_t L,
                                        const std::vector<double> &raw_tau) {
  // Check that tau is valid
  auto tau = raw_tau;
  if (tau.empty()) {
    tau.resize(L, 1.0 / L);
  } else{
    if (tau.size() != L) {
      throw std::invalid_argument("rwr::get_init_probs - tau vector must be empty or have length of L");
    }
    double tau_sum = std::accumulate(tau.begin(), tau.end(), 0.0);
    if (tau_sum != 1.0) {
      throw std::invalid_argument("rwr::get_init_probs - elements of tau must sum to 1.0");
    }
  }

  const std::size_t N = nodes_in_mp.size();
  const std::size_t NL = N * L;
  const std::size_t num_seed_vectors = seeds_vec.size();
  std::vector<double> init_probs(NL * num_seed_vectors);

  if (N == 0 || L == 0 || num_seed_vectors == 0) {
    return init_probs;
  }

  std::unordered_map<std::string, std::size_t> node_index;
  for (std::size_t i = 0; i < nodes_in_mp.size(); ++i) {
    node_index[nodes_in_mp[i]] = i;
  }

  // Loop through all seed vectors
  #pragma omp parallel for schedule(dynamic)
  for (std::size_t i_seed = 0; i_seed < seeds_vec.size(); ++i_seed) {
    double prob_sum = 0.0; // reset sum

    // Loop through all seeds in current vector
    for (const auto &seed : seeds_vec[i_seed]) {
      // Find seed in node_index
      auto it = node_index.find(seed);
      if (it != node_index.end()) {
        auto idx_in_layer = it->second;

        // fprintf(stderr, "Seed_vec: %lu - found seed %s at index %lu\n", i_seed, seed.c_str(), idx_in_layer);
        for (std::size_t i_layer = 0; i_layer < L; ++i_layer) {
          std::size_t idx = i_seed * (NL) + i_layer * N + idx_in_layer;

          init_probs[idx] = tau[i_layer];
          prob_sum += tau[i_layer];
        }
      } else {
        #pragma omp critical
        std::fprintf(stdout, "Cound not find seed %s in multiplex\n", seed.c_str());
      }
    }

    if (prob_sum == 0.0) {
      #pragma omp critical
      fprintf(stdout, "No seeds found in multiplex for seed_vec %lu\n", i_seed);
    } else if (prob_sum != 1.0) {
      for (std::size_t j = i_seed * (NL); j < (i_seed + 1) * (NL); ++j) {
        init_probs[j] /= prob_sum;
      }
    }
  }

  return init_probs;
}

// CSR_Matrix rwr::get_init_probs_sparse(const std::vector<std::vector<std::string>>& seeds_vec,
//                                       const std::vector<std::string>& nodes_in_mp,
//                                       const std::size_t L,
//                                       const std::vector<double>& raw_tau) {
//   // Check that tau is valid
//   auto tau = raw_tau;
//   if (tau.empty()) {
//     tau.resize(L, 1.0 / L);
//   } else{
//     if (tau.size() != L) {
//       throw std::invalid_argument("RWR_Utils::get_init_probs_sparse - tau vector must be empty or have length of L");
//     }
//     double tau_sum = std::accumulate(tau.begin(), tau.end(), 0.0);
//     if (tau_sum != 1.0) {
//       throw std::invalid_argument("RWR_Utils::get_init_probs_sparse - elements of tau must sum to 1.0");
//     }
//   }

//   const std::size_t N = nodes_in_mp.size();
//   const std::size_t NL = N * L;

//   std::size_t nnz = 0;
//   for (const auto& seeds : seeds_vec) {
//     nnz += seeds.size();
//   }
//   nnz *= L;

//   const std::size_t n_cols = seeds_vec.size();
//   CSR_Matrix init_probs;

//   if (N == 0 || L == 0 || nnz == 0) {
//     return init_probs;
//   }

//   init_probs.init(NL, n_cols, nnz);

//   // Determine unique seeds in 'seeds_vec' and which vector contains each unique value
//   std::unordered_map<std::string, std::vector<std::size_t>> seed_to_set_ids;
//   for (std::size_t i = 0; i < seeds_vec.size(); ++i) {
//     for (const auto& seed : seeds_vec[i]) {
//       seed_to_set_ids[seed].push_back(i);  // seed belongs to seed_vec[i]
//     }
//   }

//   // Create map between node (string) and index in 'nodes_in_mp'
//   std::unordered_map<std::string, std::size_t> node_index;
//   for (std::size_t i = 0; i < nodes_in_mp.size(); ++i) {
//     node_index[nodes_in_mp[i]] = i;
//   }

//   // Determine index of each unique seed
//   std::unordered_map<std::string, std::size_t> seed_to_node_index;
//   for (const auto& [seed, _] : seed_to_set_ids) {
//     auto it = node_index.find(seed);
//     if (it != node_index.end()) {
//       seed_to_node_index[seed] = it->second;
//     }
//   }
//   // Sort seed/index pair based on index
//   std::vector<std::pair<std::string, std::size_t>> ordered_seed_index(
//     seed_to_node_index.begin(), seed_to_node_index.end());

//   std::sort(ordered_seed_index.begin(), ordered_seed_index.end(),
//           [](const auto& a, const auto& b) {
//               return a.second < b.second;
//           });

  
//   // Loop through each layer
//   for (std::size_t l = 0; l < L; ++l) {
//     std::size_t seed_idx = 0;

//     // Loop through each node
//     for (std::size_t n = 0; n < N; ++n) {
//       std::size_t csr_row = l*L + n;

//       if (n < ordered_seed_index[seed_idx].second) {
//         init_probs.row_ptr_[csr_row + 1] = init_probs.row_ptr_[csr_row];
//       } else {
//         const auto& label = ordered_seed_index[seed_idx].first;
//         const auto& sets = seed_to_set_ids.at(label);  // get seed sets this label is in

//         init_probs.col_idx_.insert(init_probs.col_idx_.end(), sets.begin(), sets.end());
//         init_probs.values_.insert(init_probs.values_.end(), sets.size(), tau.at(l));  // binary matrix
//         init_probs.row_ptr_[csr_row + 1] = init_probs.row_ptr_[csr_row] + sets.size();

//         ++seed_idx;
//       }
//     }
//   }

//   init_probs.col_normalize();

//   return init_probs;
// }

std::vector<std::string> rwr::flatten(const std::vector<std::vector<std::string>>& input) {
  std::vector<std::string> output(input.size());

  for (std::size_t i = 0; i < input.size(); ++i) {
    auto& vec = input[i];

    std::string flat = "";
    if (vec.size() > 0) {
      flat += vec[0];

      for (std::size_t j = 1; j < vec.size(); ++j) {
        flat = flat + "_" + vec[j];
      }
    }
    output[i] = flat;
  }

  return output;
}

void rwr::remove_seeds_not_in_list(std::vector<std::vector<std::string>>& seeds, const std::vector<std::string>& x) {
  if (x.empty()) return;

  // Create a lookup set for fast membership checking
  std::unordered_set<std::string> allowed(x.begin(), x.end());

  auto is_valid_seed = [&](const std::string& seed) {
    return allowed.find(seed) != allowed.end();
  };

  // Filter each seed vector and remove empty vectors
  std::vector<std::vector<std::string>> filtered;
  filtered.reserve(seeds.size());

  for (auto& seed_vec : seeds) {
    std::vector<std::string> cleaned;
    for (const auto& seed : seed_vec) {
      if (is_valid_seed(seed)) {
        cleaned.push_back(seed);
      }
    }
    if (!cleaned.empty()) {
      filtered.emplace_back(std::move(cleaned));
    } else {
      auto str = utils::concate(seed_vec);
      fprintf(stderr, "Removed %s from seed set. Not present in list\n", str.c_str());
    }
  }

  seeds = std::move(filtered);
}

void rwr::remove_duplicate_seed_vectors(std::vector<std::vector<std::string>>& seeds) {
  std::unordered_set<std::string> seen;
  std::vector<std::vector<std::string>> unique;

  for (const auto& seed_vec : seeds) {
    std::vector<std::string> sorted_seed = seed_vec;
    std::sort(sorted_seed.begin(), sorted_seed.end());

    // Create a normalized string key
    std::string key;
    for (const auto& seed : sorted_seed) {
      key += seed + '\t';  // Use a delimiter unlikely to occur in names
    }

    if (seen.insert(key).second) {
      unique.push_back(seed_vec);  // Keep original order in the result
    } else {
      auto str = utils::concate(seed_vec);
      fprintf(stderr, "Removing %s. Duplicated in seeds.\n", str.c_str());
    }
  }

  seeds = std::move(unique);
}

void rwr::get_seed_split_for_all_workers(const int world_size,
                                         const std::size_t num_seeds,
                                         std::vector<unsigned long>& start_seed_vec,
                                         std::vector<unsigned long>& stop_seed_vec,
                                         std::vector<unsigned long>& num_encodings) {
  start_seed_vec.resize(world_size);
  stop_seed_vec.resize(world_size);
  num_encodings.resize(world_size, 0);

  for (std::size_t i = 0; i < world_size; ++i) {
    bool valid_split = split::split_tasks_among_workers(i,
                                                        world_size,
                                                        num_seeds,
                                                        start_seed_vec[i],
                                                        stop_seed_vec[i]);
    if (valid_split) {
      num_encodings[i] = stop_seed_vec[i] - start_seed_vec[i] + 1UL;
    }
  }
}

void rwr::scale_node_values(CSR_Matrix& mat,
                            const std::size_t N,
                            const std::size_t node_index,
                            const double value) {
  for (std::size_t i = 0; i < mat.n_rows(); ++i) {
    if (i % N == node_index) {
      for (int32_t idx = mat.row_ptr_[i]; idx < mat.row_ptr_[i+1]; ++idx) {
        mat.values_[idx] *= value;
      }
    } else {
      for (int32_t idx = mat.row_ptr_[i]; idx < mat.row_ptr_[i+1]; ++idx) {
        if (mat.col_idx_[idx] % N == node_index) {
          mat.values_[idx] *= value;
        }
      }
    }
  }
}
