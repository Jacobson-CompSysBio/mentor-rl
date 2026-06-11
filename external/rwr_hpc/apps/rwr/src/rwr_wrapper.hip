#include <iostream>
#include <string>
#include <vector>
#include <mpi.h>
#include <filesystem>

#include <CLI/CLI.hpp>

#include <multiplex/Multiplex.hpp>
#include <sparse/CSR_Matrix.hpp>
#include <split/split.hpp>
#include <timer/Timer.hpp>
#include "file_io/matrix_io.hpp"
#include "file_io/vector_io.hpp"
#include <parallel/parallel.hpp>
#include <rwr/rwr.hpp>
#include <rank/rank.hpp>
#include <normalize/normalize.hpp>
#include <correlate/correlate.hpp>
#include <elbow_point/elbow_point.hpp>

namespace fs = std::filesystem;

void check_and_create_path(const fs::path p) {
  // Check if p exists
  if (fs::exists(p)) {
    if (fs::is_regular_file(p)) {
      throw std::invalid_argument("rwr_wrapper - " + p.string() + " is a regulare file, not a directory");
    }
    if (!fs::is_directory(p)) {
      throw std::invalid_argument("rwr_wrapper - " + p.string() + " is another type of file (e.g. symlink, block device, etc)");
    }
  } else {
    // create directory
    fs::create_directories(p);
  }
}

int main(int argc, char** argv) {
  MPI_Init(NULL, NULL);
  {
    const int world_rank = parallel::get_comm_rank();
    const int world_size = parallel::get_comm_size();

    Timer timer;
    timer.start();

    CLI::App app{"Distributed RWR on Multiplex Networks"};

    // === CLI arguments ===
    std::string flist, mode_str, output_dir, seed_file, reduction_method, distance_metric, filter_method, runtag;
    bool no_edgelist_headers = false , no_set_ids = false, record_encodings = false, record_ranks = false, record_nodes_by_layer = false;
    bool binary_format = false;
    double r, delta, convergence_threshold;
    std::size_t n_samples_null_dist;
    
    // Path to the file containing paths to and names of each layer in the multiplex
    app.add_option("-f,--flist", flist, "Path to the file containing paths to and names of each layer in the multiplex")
      ->required();
    app.add_flag("--no_edgelist_headers", no_edgelist_headers, "Indicates that the edge lists contains no headers");
    app.add_option("-s,--seed_file", seed_file, "Tab delimited file containing seeds to initalize RWR probabilty vectors with.\n"
                                                "Seeds on a single line will be combined in a single vector.\n"
                                                "Seeds on mulitple lines will be encoded in seperate vectors.\n"
                                                "Any empty value will treat each node in the multiplex as a seed speately.")
      ->default_val("");
    app.add_flag("--no_set_ids", no_set_ids, "Indicates that the seed_file contains no set ids\n."
                                             "These ids are stored as the first value in each row.");
    app.add_option("--runtag", runtag, "string to append to output")
      ->required();
    app.add_option("-o,--output_dir", output_dir, "Output directory")
      ->default_val("./");
    app.add_option("-r,--restart", r, "Probabilty of restart during random walk")
      ->default_val(0.7)
      ->check(CLI::Range(0.0, 1.0));
    app.add_option("-d,--delta", delta, "Probabilty of jumping from a layer to any other layer in the mulitplex")
      ->default_val(0.5)
      ->check(CLI::Range(0.0, 1.0));
    app.add_option("--reduction_method", reduction_method, "The method used to reduce RWR embeddings to a single value per node")
      ->check(CLI::IsMember({"geometric", "arithmetic", "sum", "none"}))
      ->default_val("geometric");
    app.add_option("--threshold", convergence_threshold, "Used to determine when RWR converges")
      ->default_val(1e-10);
    app.add_flag("--record_encodings", record_encodings, "Indicates that the RWR encodings should be recorded");
    app.add_flag("--record_ranks", record_ranks, "Indicates that the ranked RWR encodings should be recorded");
    app.add_option("--distance_metric", distance_metric, "Metric used to calculate distance between all pairs of RWR encodings")
      ->check(CLI::IsMember({"spearman", "pearson", "dot"}))
      ->default_val("spearman");
    app.add_flag("--record_nodes_by_layer", record_nodes_by_layer, "Indicates that a file should be created recording which nodes are present in each layer");
    app.add_option("--filter_method", filter_method, "Indicates the method that should be used to filter the RWR scores before calculating the distance matrix.")
      ->check(CLI::IsMember({"","elbow"}))
      ->default_val("");
    app.add_flag("--binary_format", binary_format, "Indicates the distance matrix should be written with the data in a binary file and the meta data in a YAML file");

    // Parse user inputs and compare against CLI requirements
    try { 
      (app).parse((argc), (argv));
    } catch (const CLI::ParseError &e) {
      int cli_error = (app).exit(e);
      MPI_Abort(MPI_COMM_WORLD, 1);
      return cli_error;
    }

    std::filesystem::path output_path = output_dir;
    if (world_rank == 0) {
      check_and_create_path(output_path);
    }

    bool verbose = (world_rank == 0);
    if (verbose) {
      fprintf(stderr, "Took %lf seconds to parse input\n", timer.elapsed_wall_time());
      timer.restart();
    }

    if (verbose) {
      if (!no_edgelist_headers) {
        fprintf(stderr, "Edge lists should have headers\n");
      } else {
        fprintf(stderr, "Edge lists should not have headers\n");
      }
      if (no_set_ids) {
        fprintf(stderr, "Seeds file should have no set ids\n");
      } else {
        fprintf(stderr, "Seed files should have set ids (skip first column)\n");
      }
    }

    // fprintf(stderr, "rank%d - GPU id: %d\n", world_rank, correlate::get_device_id());

    // Read in multiplex and save local variables
    Multiplex mp(flist, !no_edgelist_headers);
    const auto node_labels = mp.get_nodes();
    const std::size_t L = mp.n_layers();
    const std::size_t N = mp.n_nodes();
    if (verbose) {
      fprintf(stderr, "Took %lf seconds to read in multiplex and save local vars\n", timer.elapsed_wall_time());
      fprintf(stderr, "Multiplex has %lu unique nodes\n", node_labels.size());
      fprintf(stderr, "Mulitplex has %lu intra-layer edges.\n", mp.get_n_intra_edges());
      timer.restart();
    }
  
    if (record_nodes_by_layer && world_rank == 0) {
      auto layer_names = mp.get_layer_names();
      std::vector<bool> nodes_by_layer = mp.get_nodes_by_layer(node_labels);
      fs::path nodes_by_layer_path = output_path / (runtag + "_nodes_by_layer.tsv");
      file_io::print_column_major_matrix(nodes_by_layer_path.string(),
                                         nodes_by_layer,
                                         layer_names.size(),
                                         node_labels.size(),
                                         layer_names,
                                         node_labels,
                                         true);
    }

    // Determine the seed set based on the provided file
    std::vector<std::vector<std::string>> seeds;
    if (seed_file.empty()) {
      rwr::get_default_seed_set(seeds, node_labels);
    } else {
      rwr::read_seeds(seeds, seed_file, !no_set_ids);
      rwr::remove_seeds_not_in_list(seeds, node_labels);
      rwr::remove_duplicate_seed_vectors(seeds);
    }
    if (verbose) {
      fprintf(stderr, "Took %lf seconds to read in seeds\n", timer.elapsed_wall_time());
      timer.restart();
    }


    //
    // Create intra-layer and inter-layer transition matrix parts
    //
    // Create the intra-layer transition matrix
    CSR_Matrix intra_tran = mp.get_intra_layer_transition_matrix(delta);
    if (verbose) {
      fprintf(stderr, "Took %lf seconds to create intra-layer transition matrix\n", timer.elapsed_wall_time());
      fprintf(stderr, "Intra-layer matrix has %lu nnz\n", intra_tran.nnz());
      timer.restart();
    }
    // Create the inter-layer transition matrix, if L>1
    std::vector<double> inter_tran = mp.get_inter_layer_transition_matrix(intra_tran, delta);
    if (verbose) {
      fprintf(stderr, "Took %lf seconds to create inter-layer transition matrix\n", timer.elapsed_wall_time());
      timer.restart();
    }

    // Determine which seed vectors each rank will encode
    std::vector<unsigned long> start_seed_vec, stop_seed_vec, num_encodings;
    rwr::get_seed_split_for_all_workers(world_size, seeds.size(),
                                        start_seed_vec, stop_seed_vec, num_encodings);
    if (verbose) {
      for (std::size_t i = 0; i < world_size; ++i) {
        fprintf(stderr, "rank_%lu: Encode %lu seeds vectors: %lu through %lu\n", i, num_encodings[i], start_seed_vec[i], stop_seed_vec[i]);
      }
    }

    // Check that at least one seed vector was assigned to worker
    if (num_encodings[world_rank] > 0) {
      // Create slice of seeds_set for current worker
      std::vector<std::vector<std::string>> worker_seeds(seeds.begin() + start_seed_vec[world_rank],
                                                        seeds.begin() + stop_seed_vec[world_rank] + 1);
      std::size_t n_encodings = worker_seeds.size();
      // fprintf(stderr, "rank%d: %lu local seed vectors\n", world_rank, worker_seeds.size());

      // vector to save encodings for current rank
      std::vector<double> encodings;

      // Encapsulte RWR Worker for auto deletion after use
      {
        // Initialize RWR worker and transfer transition matrix to GPU
        rwr::RWR_Worker rwr(N, L, intra_tran.nnz(), n_encodings, r);
        rwr.transfer_intra_transition_matrix(intra_tran.get_values(),
                                            intra_tran.get_col_idx(),
                                            intra_tran.get_row_ptr());
        rwr.transfer_inter_transition_matrix(inter_tran);

        // Encode seed vectors in a batch or micro-batch
        rwr.rwr_batch(encodings, worker_seeds, node_labels, reduction_method, {}, convergence_threshold);
      }
      
      if (verbose) {
        fprintf(stderr, "Took %lf seconds to encode seeds with rwr\n", timer.elapsed_wall_time());
        timer.restart();
      }

      // Get encoding size
      auto encoding_size = rwr::get_encoding_size(reduction_method, N, L);

      // Get number of total encodings
      auto all_seeds_flattened = rwr::flatten(seeds);
      
      if (record_encodings) {
        std::vector<double> all_encodings;
        parallel::gather(all_encodings, encodings, 0);
        if (verbose) {
          fprintf(stderr, "Gathered all encodings to rank 0\n");
        }

        if (world_rank == 0) {
          fs::path encoding_path = output_path / ( runtag + "_encodings.tsv");
          file_io::print_column_major_matrix(encoding_path.string(),
                                             all_encodings,
                                             encoding_size,
                                             all_seeds_flattened.size(),
                                             node_labels,
                                             all_seeds_flattened,
                                             16,
                                             true,
                                             true);
        }
      }

      // Ranking with ties is performed on the CPU. Check if the ranks should be recorded
      // or the selected distance matrix needs ranked data.
      if (verbose) {
        timer.restart();
      }

      if (record_ranks) {
        std::vector<double> ranks;
        rank::rank_all_vectors(
          ranks,
          encodings,
          encoding_size,
          n_encodings,
          rank::TieMethod::Average,
          rank::RankProgression::StandardCompetition,
          false
        );

        fs::path ranks_path = output_path / (runtag + "_ranks.tsv");
        std::vector<double> all_ranks;
        parallel::gather(all_ranks, ranks, 0);

        if (world_rank == 0) {
          file_io::print_column_major_matrix(ranks_path.string(),
                                            all_ranks,
                                            encoding_size,
                                            all_seeds_flattened.size(),
                                            node_labels,
                                            all_seeds_flattened,
                                            3,
                                            false,
                                            true);
        }
      }

      if (filter_method == "elbow") {
        ///
        /// Calculate the mean score for each node. Then calculate the elbow point
        /// of the rank(mean_score) vs. mean_score
        ///
        std::vector<double> local_row_sum(encoding_size, 0.0);
        std::vector<double> row_mean;


        // Calculate local row sums
        for (std::size_t col = 0; col < n_encodings; ++col) {
          for (std::size_t row = 0; row < encoding_size; ++row) {
            local_row_sum[row] += encodings[row + col * encoding_size];
          }
        }
        // Reduce across all MPI ranks
        parallel::reduce_sum_vector_vector(local_row_sum, row_mean);

        // Divide global row sum by global number of columns
        for (std::size_t row = 0; row < encoding_size; ++row) {
          row_mean[row] /= all_seeds_flattened.size();
        }

        // Rank the row mean score
        std::vector<double> row_ranks(row_mean.size());
        rank::rank_all_vectors(
          row_ranks,
          row_mean,
          row_mean.size(),
          1,
          rank::TieMethod::Average,
          rank::RankProgression::StandardCompetition,
          false
        );

        // Find elbow point of the mean row scores
        auto elb = elbow_point::elbow_point(row_ranks, row_mean);
        double elbow_x = elb.first;
        
        // Calculate ranks of local data
        std::vector<double> ranks;
        rank::rank_all_vectors(
          ranks,
          encodings,
          encoding_size,
          n_encodings,
          rank::TieMethod::Average,
          rank::RankProgression::StandardCompetition,
          false
        );

        // Calculate rows to keep based on local ranks
        std::vector<char> keep(encoding_size);
        for (std::size_t i = 0; i < encoding_size; ++i) {
          char keep_row = 0;

          for (std::size_t j = 0; j < n_encodings; ++j) {
            if (ranks[j * encoding_size + i] < elbow_x) {
              keep_row = 1;
            }
          }

          keep[i] = keep_row;
        }

        // Bit-wise OR of all local keep vectors
        std::vector<char> global_keep;
        parallel::reduce_or_vector_char(keep, global_keep);

        // Count rows to keep
        std::size_t filtered_encoding_size = 0;
        for (const auto& k : global_keep) {
          if (k) ++filtered_encoding_size;
        }
        fprintf(stderr, "Elbow point filtering kept %lu of %lu elements per vector\n", filtered_encoding_size, encoding_size);

        // Filter encodings
        std::vector<double> filtered_encodings;
        filtered_encodings.reserve(n_encodings * filtered_encoding_size);
        for (std::size_t j = 0; j < n_encodings; ++j) {
          for (std::size_t i = 0; i < encoding_size; ++i) {
            if (global_keep[i]) {
              std::size_t idx = i + j * encoding_size;
              filtered_encodings.push_back(encodings[idx]);
            }
          }
        }

        std::swap(encodings, filtered_encodings);
      }
      encoding_size = encodings.size() / n_encodings;
      
      std::vector<double> all_encodings;
      parallel::gather(all_encodings, encodings, 0);

      if (world_rank == 0) {
        timer.restart();

        std::vector<double> dist_matrix(all_seeds_flattened.size() * all_seeds_flattened.size());

        if (distance_metric.compare("spearman") == 0) {
          correlate::spearman::local_distance(dist_matrix, all_encodings, encoding_size, all_seeds_flattened.size());
          // correlate::spearman::local(dist_matrix, all_encodings, encoding_size, all_seeds_flattened.size());
        } else if (distance_metric.compare("pearson") == 0) {
          correlate::pearson::local_distance(dist_matrix, all_encodings, encoding_size, all_seeds_flattened.size(), true, true);
        } else if (distance_metric.compare("dot") == 0) {
          correlate::dot::local(dist_matrix, all_encodings, encoding_size, all_seeds_flattened.size());
        } else {
          throw std::runtime_error("unknow distance metric");
        }
        
        fprintf(stderr, "Took %lf seconds to calculate distance matrix\n", timer.elapsed_wall_time());
        timer.restart();
        
        if (binary_format) {
          fs::path dist_matrix_path = output_path / ( runtag + "_"  + distance_metric + "_dist_matrix.bin");
          fs::path yaml_path = output_path / ( runtag + "_"  + distance_metric + "_dist_matrix.yaml");
          
          file_io::write_column_major_matrix_with_yaml(
            dist_matrix_path.string(),
            yaml_path.string(),
            dist_matrix,
            all_seeds_flattened.size(),
            all_seeds_flattened.size(),
            all_seeds_flattened,
            all_seeds_flattened
          );
        } else {
          fs::path dist_matrix_path = output_path / ( runtag + "_"  + distance_metric + "_dist_matrix.tsv");

          fprintf(stderr, "Writing out to %s\n", dist_matrix_path.string().c_str());
          // Write to output file
          file_io::print_column_major_distance_matrix(
            dist_matrix_path.string(),
            dist_matrix,
            all_seeds_flattened.size(),
            all_seeds_flattened,
            16,
            false,
            file_io::DIST_MATRIX_MODE::LOWER
          );
          fprintf(stderr, "Took %lf seconds to write distance matrix\n", timer.elapsed_wall_time());
        }
      }
    }
  }
  
  MPI_Finalize();
  return 0;
}
