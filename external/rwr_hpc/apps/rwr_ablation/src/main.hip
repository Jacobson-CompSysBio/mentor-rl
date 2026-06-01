#include <mpi.h>

#include <CLI/CLI.hpp>

#include <parallel/parallel.hpp>
#include <multiplex/Multiplex.hpp>
#include <sparse/CSR_Matrix.hpp>
#include <rwr/rwr.hpp>
#include <rank/rank.hpp>
#include <normalize/standard.hpp>
#include <correlate/correlate.hpp>
#include <file_io/matrix_io.hpp>
#include <timer/Timer.hpp>

int main(int argc, char* argv[]) {
  MPI_Init(NULL, NULL);
  {
    int world_rank = parallel::get_comm_rank();
    int world_size = parallel::get_comm_size();

    Timer timer;
    timer.start();

    CLI::App app{"Distributed layer ablation using RWR embeddings on multiplex networks"};

    // === CLI arguments ===
    std::string flist, output_dir, reduction_method, distance_metric, seed_file;
    bool no_edgelist_headers = false, no_set_ids = false;
    double r, delta, convergence_threshold;

    app.add_option("-f,--flist", flist, "Path to the file containing paths to and names of each layer in the multiplex")
      ->required();
    app.add_flag("--no_edgelist_headers", no_edgelist_headers, "Indicates that the edge lists contains no headers");
    app.add_option("-s,--seed_file", seed_file, "Tab delimited file containing seeds to initalize RWR probabilty vectors with.\n"
                                                "Seeds on a single line will be combined in a single vector.\n"
                                                "Seeds on mulitple lines will be encoded in seperate vectors.\n"
                                                "Any empty value will treat each node in the multiplex as a seed separately.")
      ->default_val("");
    app.add_flag("--no_set_ids", no_set_ids, "Indicates that the pertubation_file and seed_file contains no set ids\n."
                                             "These ids are stored as the first value in each row.");
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
    app.add_option("--distance_metric", distance_metric, "Metric used to calculate distance between all pairs of RWR encodings")
      ->check(CLI::IsMember({"spearman", "pearson", "cos"}))
      ->default_val("spearman");

    // Parse user inputs and compare against CLI requirements
    try { 
      (app).parse((argc), (argv));
    } catch (const CLI::ParseError &e) {
      int cli_error = (app).exit(e);
      MPI_Abort(MPI_COMM_WORLD, 1);
      return cli_error;
    }

    if (world_rank == 0) {
      // Check if output_dir and temp_dir exist
      std::filesystem::path output_path = output_dir;
      if (!std::filesystem::is_directory(output_path)) {
        std::filesystem::create_directories(output_path);
      }
    }

    bool verbose = (world_rank == 0);
    if (verbose) {
      fprintf(stderr, "Took %lf seconds to parse input\n", timer.elapsed_wall_time());
      timer.restart();
    }

    // Read in multiplex and save local variables
    Multiplex mp(flist, !no_edgelist_headers);
    const auto node_labels = mp.get_nodes();
    const std::size_t N = mp.n_nodes();
    const std::size_t L = mp.n_layers();
    if (verbose) {
      fprintf(stderr, "Took %lf seconds to read in multiplex and save local vars\n", timer.elapsed_wall_time());
      fprintf(stderr, "Multiplex has %lu unique nodes\n", node_labels.size());
      timer.restart();
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
    auto all_seeds_flattened = rwr::flatten(seeds);

    if (verbose) {
      fprintf(stderr, "Calculate distance matrix with %lu seeds\n", all_seeds_flattened.size());
    }

    // Determine which seed vectors each rank will encode
    std::vector<unsigned long> start_seed_vec, stop_seed_vec, num_encodings;
    rwr::get_seed_split_for_all_workers(world_size, seeds.size(),
                                        start_seed_vec, stop_seed_vec, num_encodings);

    // Check that at least one seed vector was assigned to worker
    if (num_encodings[world_rank] > 0) {
      // Create slice of seeds_set for current worker
      std::vector<std::vector<std::string>> worker_seeds(seeds.begin() + start_seed_vec[world_rank],
                                                         seeds.begin() + stop_seed_vec[world_rank] + 1);
      // Save number of encodings 
      std::size_t n_encodings = worker_seeds.size();
      std::size_t encoding_size = rwr::get_encoding_size(reduction_method, N, L);

      // vector to save encodings for current rank
      std::vector<double> base_encodings, ablation_encodings;

      // Calculate base encodings (no ablation)
      timer.restart();
      CSR_Matrix intra_tran = mp.get_intra_layer_transition_matrix(delta);
      {
        // Create intra-layer and inter-layer transition matrix parts
        if (verbose) {
          fprintf(stderr, "Took %lf seconds to create intra-layer transition matrix\n", timer.elapsed_wall_time());
          fprintf(stderr, "Intra-layer matrix has %lu nnz\n", intra_tran.nnz());
          timer.restart();
        }

        std::vector<double> inter_tran = mp.get_inter_layer_transition_matrix(intra_tran, delta);
        if (verbose) {
          fprintf(stderr, "Took %lf seconds to create inter-layer transition matrix\n", timer.elapsed_wall_time());
          timer.restart();
        }

        // Initialize RWR worker and transfer transition matrix to GPU
        rwr::RWR_Worker rwr(N, L, intra_tran.nnz(), n_encodings, r);
        rwr.transfer_intra_transition_matrix(intra_tran.get_values(),
                                             intra_tran.get_col_idx(),
                                             intra_tran.get_row_ptr());
        rwr.transfer_inter_transition_matrix(inter_tran);

        // Encode seed vectors in a batch or micro-batch
        rwr.rwr_batch(base_encodings, worker_seeds, node_labels, reduction_method, {}, convergence_threshold);
      }

      if (verbose) {
        fprintf(stderr, "Finished base encoding - took %lf seconds\n", timer.elapsed_wall_time());
        timer.restart();
      }
      
      if (distance_metric.compare("spearman") == 0) {
        // Calculate ranks
        std::vector<double> base_ranks;
        rank::rank_all_vectors(
          base_ranks,
          base_encodings,
          encoding_size,
          n_encodings,
          rank::TieMethod::Average,
          rank::RankProgression::StandardCompetition,
          false
        );

        // Standardize base_ranks
        normalize::standard::fit_transform(base_ranks, encoding_size, n_encodings, false);

        std::swap(base_encodings, base_ranks);

        if (verbose) {
          fprintf(stderr, "Took %lf seconds to rank and standardize encodings\n", timer.elapsed_wall_time());
          timer.restart();
        }

      } else if (distance_metric.compare("pearson") == 0) {
        // Standardize base_encodings
        normalize::standard::fit_transform(base_encodings, encoding_size, n_encodings, false);

        if (verbose) {
          fprintf(stderr, "Took %lf seconds to standardize encodings\n", timer.elapsed_wall_time());
          timer.restart();
        }
      }

      // Initalize storage for all distance calculation on this worker
      std::vector<double> ablation_distance(n_encodings * L, 1.0);
      std::size_t ablation_distance_offset = 0;

      // Loop through all layers
      for (std::size_t l = 0; l < L; ++l) {
        std::vector<bool> layer_list(L, true);
        layer_list[l] = false; // remove the l-th layer

        if (verbose) {
          fprintf(stderr, "Ablating layer %lu\n", l);
        }

        // Calculate ablation encodings
        {
          // Create intra-layer and inter-layer transition matrix parts
          // CSR_Matrix ablation_intra_tran = mp.get_intra_layer_transition_matrix(delta, node_labels, layer_list);
          CSR_Matrix ablation_intra_tran;
          std::vector<std::size_t> idx_to_remove;
          idx_to_remove.reserve(N);
          for (std::size_t i = l * N; i < N * (l + 1); ++i) {
            idx_to_remove.push_back(i);
          }
          intra_tran.remove_rows_cols(ablation_intra_tran, idx_to_remove, idx_to_remove);
          std::vector<double> ablation_inter_tran = mp.get_inter_layer_transition_matrix(ablation_intra_tran, delta, node_labels, layer_list);

          // Initialize RWR worker and transfer transition matrix to GPU
          rwr::RWR_Worker rwr(N, L-1, ablation_intra_tran.nnz(), n_encodings, r);
          rwr.transfer_intra_transition_matrix(ablation_intra_tran.get_values(),
                                               ablation_intra_tran.get_col_idx(),
                                               ablation_intra_tran.get_row_ptr());
          rwr.transfer_inter_transition_matrix(ablation_inter_tran);

          // Encode seed vectors in a batch or micro-batch
          timer.restart();
          rwr.rwr_batch(ablation_encodings, worker_seeds, node_labels, reduction_method, {}, convergence_threshold);
        }
        if (verbose) {
          fprintf(stderr, "Took %lf seconds to encode seeds with rwr\n", timer.elapsed_wall_time());
          timer.restart();
        }

        // Calculate distance between vector 'i' in base_encodings and vector 'i' ablation_encodings
        if (distance_metric.compare("spearman") == 0) {
          // Calculate ranks
          std::vector<double> ablation_ranks;
          rank::rank_all_vectors(
            ablation_ranks,
            ablation_encodings,
            encoding_size,
            n_encodings,
            rank::TieMethod::Average,
            rank::RankProgression::StandardCompetition,
            false
          );

          // Standardize ablation_ranks
          normalize::standard::fit_transform(ablation_ranks, encoding_size, n_encodings, false);

          correlate::dot::local_corresponding_columns(ablation_distance, ablation_distance_offset, base_encodings, ablation_ranks, encoding_size, n_encodings, -1.0 / (encoding_size - 1), 1.0);
        } else if (distance_metric.compare("pearson") == 0) {
          // Standardize base_encodings
          normalize::standard::fit_transform(ablation_encodings, encoding_size, n_encodings, false);

          correlate::dot::local_corresponding_columns(ablation_distance, ablation_distance_offset, base_encodings, ablation_encodings, encoding_size, n_encodings, -1.0 / (encoding_size - 1), 1.0);
        } else {
          correlate::cos::local_corresponding_columns(ablation_distance, ablation_distance_offset, base_encodings, ablation_encodings, encoding_size, n_encodings, -1.0, 1.0);
        }

        ablation_distance_offset += n_encodings;
      }

      // Gather ablation_distance values from all workers
      std::vector<double> all_ablation_distance;
      parallel::gather_and_row_join_column_major(all_ablation_distance, ablation_distance, L);

      // Record distance matrix
      if (world_rank == 0) {
        std::string dist_matrix_file_name = output_dir + distance_metric + "_ablation_distance_matrix.tsv";

        fprintf(stderr, "Writing out to %s\n", dist_matrix_file_name.c_str());
        file_io::print_column_major_matrix(dist_matrix_file_name,
                                           all_ablation_distance,
                                           all_seeds_flattened.size(),
                                           L,
                                           all_seeds_flattened,
                                           mp.get_layer_names(),
                                           16,
                                           false);
      }
    }
  }
  MPI_Finalize();
  return 0;
}