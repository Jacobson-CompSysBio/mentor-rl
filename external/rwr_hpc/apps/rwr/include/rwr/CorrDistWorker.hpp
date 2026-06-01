#pragma once

#include <string>
#include <vector>

bool needs_rank_data(const std::string& method);

void calculate_pairwise_distance(std::vector<double>& dist_matrix,
                                 const std::vector<double>& data_matrix,
                                 const std::size_t M,
                                 const std::size_t N,
                                 const std::string& method,
                                 const bool gpu = true,
                                 const bool mpi = true);

void calculate_pairwise_spearman_distance(std::vector<double>& dist_matrix,
                                          const std::vector<double>& data_matrix,
                                          const std::size_t M,
                                          const std::size_t N,
                                          const bool gpu = true,
                                          const bool mpi = true);

void calculate_pairwise_pearson_distance(std::vector<double>& dist_matrix,
                                         const std::vector<double>& data_matrix,
                                         const std::size_t M,
                                         const std::size_t N,
                                         const std::string& method,
                                         const bool gpu = true,
                                         const bool mpi = true);
