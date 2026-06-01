#include <iostream>
#include <string>
#include <vector>
#include <mpi.h>
#include <filesystem>

#include <CLI/CLI.hpp>

#include <parallel/parallel.hpp>
#include <multiplex/Multiplex.hpp>
#include <sparse/CSR_Matrix.hpp>
#include <split/split.hpp>
#include <timer/Timer.hpp>
#include <stats/stats.hpp>
#include <file_io/matrix_io.hpp>
#include <file_io/vector_io.hpp>
#include <rank/rank.hpp>
#include <rwr/rwr.hpp>
#include <sort/sort.hpp>

namespace fs = std::filesystem;
void check_and_create_path(const fs::path p) {
  // Check if p exists
  if (fs::exists(p)) {
    if (fs::is_regular_file(p)) {
      throw std::invalid_argument("grin - " + p.string() + " is a regulare file, not a directory");
    }
    if (!fs::is_directory(p)) {
      throw std::invalid_argument("grin - " + p.string() + " is another type of file (e.g. symlink, block device, etc)");
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

    CLI::App app{"Distributed GRIN filtering"};

    // CLI arguments
    std::string flist, runtag, output_dir, temp_dir, seed_file, reduction_method, distance_metric;
    bool no_edgelist_headers = false, no_set_ids = false, record_encodings = false, record_ranks = false;
    double r, delta, convergence_threshold;
    std::size_t n_samples_null_dist;
    uint64_t seed;
    
    // Path to the file containing paths to and names of each layer in the multiplex
    app.add_option("-f,--flist", flist, "Path to the file containing paths to and names of each layer in the multiplex")
      ->required();
    app.add_flag("--no_edgelist_headers", no_edgelist_headers, "Indicates that the edge lists contains no headers");
    app.add_option("--seed_file", seed_file, "Tab delimited file containing seeds to initalize RWR probabilty vectors with.\n"
                                            "Seeds on a single line will be combined in a single vector.\n"
                                            "Seeds on mulitple lines will be encoded in seperate vectors.")
      ->required();
    app.add_flag("--no_set_ids", no_set_ids, "Indicates that the seed_file contains no set ids\n."
                                             "These ids are stored as the first value in each row.");
    app.add_option("--runtag", runtag, "string to append to output")
      ->required();
    app.add_option("-o,--output_dir", output_dir, "Output directory")
      ->default_val("./");
    app.add_option("--temp_dir", temp_dir, "Directory for any intermediate files")
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
    app.add_option("--n_samples_null_dist", n_samples_null_dist, "Number of random seeds sets to use to create null distribution.")
      ->default_val(100)
      ->check(CLI::Range(10000));
    app.add_option("--seed", seed, "Value used to seed random number generator for null distribution")
      ->default_val(42UL);                 

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
    
    // Read in multiplex and save local variables
    Multiplex mp(flist, !no_edgelist_headers);
    const auto node_labels = mp.get_nodes();
    const std::size_t L = mp.n_layers();
    const std::size_t N = mp.n_nodes();

    // Read in base seeds and generate leave-one-out seed and seeds sets
    std::vector<std::vector<std::string>> seeds;
    std::vector<std::string> base_seeds, loo_seeds;
    std::filesystem::path tmp_path = temp_dir;
    std::filesystem::path loo_seeds_path = tmp_path / "loo_seeds.txt";
    std::filesystem::path grin_seeds_path = tmp_path / "grin_seeds.txt";

    std::vector<uint32_t> idx_buffer;
    uint64_t buffer_size, n_seed_vectors;

    if (world_rank == 0) {
      if (seed_file.empty()) {
        throw std::runtime_error("must provide seed_file when in grin mode");
      }
      std::vector<std::vector<std::string>> base_seeds_raw;
      rwr::read_seeds(base_seeds_raw, seed_file, !no_set_ids);
      rwr::remove_seeds_not_in_list(base_seeds_raw, node_labels);
      rwr::remove_duplicate_seed_vectors(base_seeds_raw);

      base_seeds = rwr::flatten(base_seeds_raw);
      rwr::get_grin_loo_seeds(seeds, loo_seeds, base_seeds, node_labels, n_samples_null_dist, seed);

      fprintf(stderr, "Num seeds: %lu\n", seeds.size());
      fprintf(stderr, "Num loo_seeds: %lu\n", loo_seeds.size());

      // Create map for each unique node in the multiplex to an index
      std::unordered_map<std::string, uint32_t> node_label_dict;
      node_label_dict.reserve(node_labels.size());
      for (uint32_t i = 0; i < node_labels.size(); ++i) {
        node_label_dict.emplace(node_labels[i], i);
      }

      auto map_strings_to_idx = 
      [&node_label_dict](std::vector<std::uint32_t>& out,
                         const std::vector<std::string>& values)
      {
        const std::size_t n = values.size();
        out.resize(n);

        for (std::size_t i = 0; i < n; ++i) {
          auto it = node_label_dict.find(values[i]);
          if (it == node_label_dict.end()) {
            throw std::runtime_error("String not found in dictionary");
          }
          out[i] = it->second;
        }
      };

      // Convert LOO seeds to uint32_t
      map_strings_to_idx(idx_buffer, loo_seeds);

      // Broadcast the number of loo_seeds
      buffer_size = idx_buffer.size();
      MPI_Bcast(&buffer_size, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);

      // Broadcast seeds
      MPI_Bcast(idx_buffer.data(), buffer_size, MPI_UINT32_T, 0, MPI_COMM_WORLD);

      // Broadcast number of seed vectors
      n_seed_vectors = seeds.size();
      MPI_Bcast(&n_seed_vectors, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);

      // Loop through each seed vector index
      for (std::size_t j = 0; j < n_seed_vectors; ++j) {
        // Convert seed vector to uint32_t
        map_strings_to_idx(idx_buffer, seeds[j]);

        // Broadcast the number of elements in seed vector
        buffer_size = idx_buffer.size();
        MPI_Bcast(&buffer_size, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);

        // Broadcast seed vector
        MPI_Bcast(idx_buffer.data(), buffer_size, MPI_UINT32_T, 0, MPI_COMM_WORLD);
      }
      
    } else {
      // Receive number of loo_seeds
      MPI_Bcast(&buffer_size, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);

      // Allocate buffer
      idx_buffer.resize(buffer_size);
      loo_seeds.resize(buffer_size);

      // Receive loo_seeds_index
      MPI_Bcast(idx_buffer.data(), buffer_size, MPI_UINT32_T, 0, MPI_COMM_WORLD);

      // Convert index to std::string
      for (std::size_t i = 0; i < buffer_size; ++i) {
        loo_seeds[i] = node_labels[idx_buffer[i]];
      }

      // Receive numbeer of seeds
      MPI_Bcast(&n_seed_vectors, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);

      seeds.resize(n_seed_vectors);

      // Loop through each seed vectors
      for (std::size_t j = 0; j < n_seed_vectors; ++j) {
        // Receive number of seeds in vetor
        MPI_Bcast(&buffer_size, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);

        // Allocate buffer
        idx_buffer.resize(buffer_size);
        seeds[j].resize(buffer_size);

        // Receive seeds in vector
        MPI_Bcast(idx_buffer.data(), buffer_size, MPI_UINT32_T, 0, MPI_COMM_WORLD);

        // Convert index to std::string
        for (std::size_t i = 0; i < buffer_size; ++i) {
          seeds[j][i] = node_labels[idx_buffer[i]];
        }
      }
    }

    // MPI_Barrier(MPI_COMM_WORLD);
      
    // if (world_rank != 0) {
    //   loo_seeds = file_io::read_vector<std::string>(loo_seeds_path.string());
    //   rwr::read_seeds(seeds, grin_seeds_path.string(), false);
    // }

    // Make sure all ranks have converted index to strings
    MPI_Barrier(MPI_COMM_WORLD);
      
    if (verbose) {
      fprintf(stderr, "Took %lf seconds to read in seeds\n", timer.elapsed_cpu_time());
      timer.restart();
    }

    // Create intra-layer and inter-layer transition matrix parts
    CSR_Matrix intra_tran = mp.get_intra_layer_transition_matrix(delta);
    std::vector<double> inter_tran = mp.get_inter_layer_transition_matrix(intra_tran, delta);

    // Determine which seed vectors each rank will encode
    std::vector<unsigned long> start_seed_vec, stop_seed_vec, num_encodings;
    rwr::get_seed_split_for_all_workers(world_size, seeds.size(),
                                        start_seed_vec, stop_seed_vec, num_encodings);

    // Check that at least one seed vector was assigned to worker
    if (num_encodings[world_rank] > 1) {
      // Create slice of seeds_set for current worker
      std::vector<std::vector<std::string>> worker_seeds(seeds.begin() + start_seed_vec[world_rank],
                                                         seeds.begin() + stop_seed_vec[world_rank] + 1);
      std::vector<std::string> worker_loo_seeds(loo_seeds.begin() + start_seed_vec[world_rank],
                                                loo_seeds.begin() + stop_seed_vec[world_rank] + 1);

      // Save encoding size and number of encodings to local vars
      const auto encoding_size = rwr::get_encoding_size(reduction_method, N, L);
      const std::size_t n_encodings = worker_seeds.size();

      // vector to save encodings for current worker
      std::vector<double> encodings;
      {
        // Initialize RWR worker and transfer transition matrix to GPU
        rwr::RWR_Worker rwr(N, L, intra_tran.nnz(), worker_seeds.size(), r);
        rwr.transfer_intra_transition_matrix(intra_tran.get_values(),
                                            intra_tran.get_col_idx(),
                                            intra_tran.get_row_ptr());
        rwr.transfer_inter_transition_matrix(inter_tran);

        fprintf(stderr, "Allocated GPU memory\n");

        // Encode seed vectors in a batch or micro-batch
        rwr.rwr_batch(encodings, worker_seeds, node_labels, reduction_method, {}, convergence_threshold);
      }

      if (verbose) {
        fprintf(stderr, "Finished RWR\n");
      }

      if (record_encodings) {
        std::vector<double> all_encodings;
        parallel::gather(all_encodings, encodings, 0);
        if (verbose) {
          fprintf(stderr, "Gathered all encodings to rank 0\n");
        }

        if (world_rank == 0) {
          timer.restart();
          fs::path encoding_path = output_path / (runtag + "_encodings.tsv");
          file_io::print_column_major_matrix(encoding_path.string(),
                                             all_encodings,
                                             encoding_size,
                                             loo_seeds.size(),
                                             node_labels,
                                             loo_seeds,
                                             16,
                                             true,
                                             true);
          fprintf(stderr, "Took %lf seconds to record encodings\n", timer.elapsed_cpu_time());
        }
      }

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

      if (verbose) {
        fprintf(stderr, "Calculating ranks for GRIN++\n");
      }
      
      if (record_ranks) {
        fs::path rank_path = output_path / ( runtag + "_ranks.tsv");
        std::vector<double> all_ranks;
        parallel::gather(all_ranks, ranks, 0);

        if (world_rank == 0) {
          fprintf(stderr, "Recording ranks for GRIN++\n");
          file_io::print_column_major_matrix(rank_path.string(),
                                            all_ranks,
                                            encoding_size,
                                            loo_seeds.size(),
                                            node_labels,
                                            loo_seeds,
                                            3,
                                            false,
                                            true);
        }
      }

      // Get the ranks of the loo_seeds assign to this rank
      std::vector<double> loo_ranks(n_encodings);
      for (std::size_t i = 0; i < n_encodings; ++i) {
        auto itr = std::find(node_labels.begin(), node_labels.end(), worker_loo_seeds[i]);
        if (itr == node_labels.end()) {
          throw std::runtime_error("Could not find loo_seed in node_labels: " + worker_loo_seeds[i]);
        }
        const std::size_t idx = std::distance(node_labels.begin(), itr);
        const std::size_t offset = encoding_size * i;
        loo_ranks[i] = ranks[offset + idx];
      }

      // Gather all of the loo_ranks to rank0
      std::vector<double> all_loo_ranks;
      parallel::gather(all_loo_ranks, loo_ranks, 0);

      if (verbose) {
        fprintf(stderr, "Gathered all ranks to root\n");
      }

      if (world_rank == 0) {
        const std::size_t n_base_seeds = base_seeds.size();

        fprintf(stderr, "Num elements in all_loo_ranks matrix: %lu\n", all_loo_ranks.size());
        fprintf(stderr, "n_base_seeds: %lu\n", n_base_seeds);
        fprintf(stderr, "n_samples_null_dist: %lu\n", n_samples_null_dist);

        // Save the ranks of the provided seeds
        std::vector<double> base_loo_ranks(n_base_seeds);
        for (std::size_t i = 0; i < n_base_seeds; ++i) {
          base_loo_ranks[i] = all_loo_ranks[i];
        }

        // Sort base_loo_ranks and base_seeds
        sort::sort_two_vectors(base_loo_ranks, base_seeds);

        fs::path base_ranks_path = output_path / (runtag + "_gene_ranks.tsv");
        file_io::print_column_major_matrix(base_ranks_path.string(),
                                          base_loo_ranks,
                                          n_base_seeds,
                                          1,
                                          base_seeds,
                                          {"rank"},
                                          3,
                                          false,
                                          false);

        fprintf(stderr, "Computed LOO ranks for input set\n");

        // Sort all_loo_ranks before calculating median values
        sort::sort_columns_in_column_major_matrix(all_loo_ranks, n_base_seeds, n_samples_null_dist+1);
        if (world_rank == 0) {
          fs::path all_loo_ranks_paths = output_path / (runtag + "_all_loo_ranks.tsv");
          file_io::print_column_major_matrix(all_loo_ranks_paths.string(),
                                          all_loo_ranks,
                                          base_seeds.size(),
                                          n_samples_null_dist+1,
                                          {},
                                          {},
                                          3,
                                          false,
                                          false);
      }

        // Get the rank of the null distribution seeds and calculate the median value for each position
        std::vector<double> rank_buffer(n_samples_null_dist);
        std::vector<double> null_median_ranks(n_base_seeds);
        for (std::size_t pos = 0; pos < n_base_seeds; ++pos) {
          for (std::size_t j = 0; j < n_samples_null_dist; ++j) {
            const std::size_t idx = n_base_seeds * (1 + j) + pos;
            rank_buffer[j] = all_loo_ranks[idx];
          }
          null_median_ranks[pos] = stats::compute_median(rank_buffer, true);
        }

        fprintf(stderr, "Computed LOO ranks for null distrobution\n");


        std::vector<std::string> null_labels;
        for (std::size_t i = 0; i < n_base_seeds; ++i) {
          std::string label = "null_rank_pos" + std::to_string(i);
          null_labels.push_back(label);
        }

        fs::path null_ranks_path = output_path / (runtag + "_null_ranks.tsv");
        file_io::print_column_major_matrix(null_ranks_path.string(),
                                          null_median_ranks,
                                          n_base_seeds,
                                          1,
                                          null_labels,
                                          {"rank"},
                                          3,
                                          false,
                                          false);

      }
    }

    if (verbose) {
      fprintf(stderr, "Finished\n");
    }
  }
  
  MPI_Finalize();
  return 0;
}
