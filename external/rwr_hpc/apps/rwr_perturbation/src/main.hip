#include <mpi.h>

#include <iterator>

#include <CLI/CLI.hpp>
#include <parallel/parallel.hpp>
#include <multiplex/Multiplex.hpp>
#include <sparse/CSR_Matrix.hpp>
#include <rwr/rwr.hpp>
#include <rank/rank.hpp>
#include <normalize/standard.hpp>
#include <correlate/correlate.hpp>
#include <file_io/matrix_io.hpp>

int main(int argc, char* argv[]) {
  MPI_Init(NULL, NULL);
  {
    CLI::App app{"Distributed node perturbation using RWR embeddings on multiplex networks"};

    // === CLI arguments ===
    std::string flist, output_dir, reduction_method, distance_metric, seed_file, pert_file;
    bool no_edgelist_headers = false, no_set_ids = false;
    double r, delta, convergence_threshold;

    app.add_option("-f,--flist", flist, "Path to the file containing paths to and names of each layer in the multiplex")
      ->required();
    app.add_flag("--no_edgelist_headers", no_edgelist_headers, "Indicates that the edge lists contains no headers");
    app.add_option("-p,--pertubation_file", pert_file, "Tab delimited file containing nodes to perturbate.\n"
                                                       "Nodes on a single line will be perturbed together.\n"
                                                       "Nodes on mulitple lines will be perturbed separately.\n"
                                                       "Any empty value will perturbe each node in the multiplex separately.")
      ->default_val("");
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
    app.add_option("--distance_metric", distance_metric, "Metric used to calculate distance between base encoding and perburbation encoding for all seeds")
      ->check(CLI::IsMember({"spearman", "pearson", "dot", "cos"}))
      ->default_val("spearman");

    // Parse user inputs and compare against CLI requirements
    try { 
      (app).parse((argc), (argv));
    } catch (const CLI::ParseError &e) {
      int cli_error = (app).exit(e);
      MPI_Abort(MPI_COMM_WORLD, 1);
      return cli_error;
    }

    int world_rank = parallel::get_comm_rank();
    int world_size = parallel::get_comm_size();

    // Read in multiplex and save local variables
    Multiplex mp(flist, !no_edgelist_headers);
    const auto node_labels = mp.get_nodes();
    const std::size_t L = mp.n_layers();
    const std::size_t N = mp.n_nodes();

    // Determine the seed set based on the provided file
    std::vector<std::vector<std::string>> seeds;
    if (seed_file.empty()) {
      rwr::get_default_seed_set(seeds, node_labels);
    } else {
      rwr::read_seeds(seeds, seed_file, !no_set_ids);
      rwr::remove_seeds_not_in_list(seeds, node_labels);
      rwr::remove_duplicate_seed_vectors(seeds);
    }

    // Determine the perturbation node set based on the provided file
    std::vector<std::vector<std::string>> pert_node_sets;
    if (pert_file.empty()) {
      rwr::get_default_seed_set(pert_node_sets, node_labels);
    } else {
      rwr::read_seeds(pert_node_sets, pert_file, !no_set_ids);
      rwr::remove_seeds_not_in_list(pert_node_sets, node_labels);
      rwr::remove_duplicate_seed_vectors(pert_node_sets);
    }

    // Create intra-layer and inter-layer transition matrix parts
    CSR_Matrix intra_tran = mp.get_intra_layer_transition_matrix(delta);
    std::vector<double> inter_tran = mp.get_inter_layer_transition_matrix(intra_tran, delta);

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
      std::vector<std::string> worker_seeds_flattened = rwr::flatten(worker_seeds);
      auto all_seeds_flattened = rwr::flatten(seeds);
      auto pert_nodes_flattened = rwr::flatten(pert_node_sets);

      if (world_rank == 0)
        fprintf(stderr, "rank%d is encoding %lu nodes\n", world_rank, n_encodings);

      // vector to save encodings for current rank
      std::vector<double> base_encodings;

      // Calculate base encodings (no perturbation)
      {
        // Initialize RWR worker and transfer transition matrix to GPU
        rwr::RWR_Worker rwr(N, L, intra_tran.nnz(), n_encodings, r);
        rwr.transfer_intra_transition_matrix(intra_tran.get_values(),
                                             intra_tran.get_col_idx(),
                                             intra_tran.get_row_ptr());
        rwr.transfer_inter_transition_matrix(inter_tran);

        // Encode seed vectors in a batch or micro-batch
        rwr.rwr_batch(base_encodings, worker_seeds, node_labels, reduction_method, {}, convergence_threshold);
      }

      if (world_rank == 0)
        fprintf(stderr, "Finished base encoding\n");

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

        if (world_rank == 0)
          fprintf(stderr, "ranked and standardized base encodings\n");

      } else if (distance_metric.compare("pearson") == 0) {
        // Standardize base_encodings
        normalize::standard::fit_transform(base_encodings, encoding_size, n_encodings, false);

        if (world_rank == 0)
          fprintf(stderr, "standardized base encodings\n");
      }

      
      
      // Save transition matrix
      std::vector<double> intra_tran_copy = intra_tran.values_;
      // if (world_rank == 0)
      //   fprintf(stderr, "saved intra_tran values\n");

      // Initialize RWR worker
      rwr::RWR_Worker rwr(N, L, intra_tran.nnz(), n_encodings, r);
      if (world_rank == 0)
        fprintf(stderr, "Reinitialized RWR Worker\n");

      std::vector<double> pert_distance(n_encodings * pert_node_sets.size(), 1.0);
      std::size_t pert_distance_offest = 0;
      // Loop through all nodes to perturb
      for (const auto& pert_nodes : pert_node_sets) {
        if (world_rank == 0)
          fprintf(stderr, "pert offset: %lu\n", pert_distance_offest);
        
        // Reset the transition matrix values
        std::copy(intra_tran_copy.begin(), intra_tran_copy.end(), intra_tran.values_.begin());

        if (world_rank == 0)
          fprintf(stderr, "reset intra_tran values\n");

        for (auto& node_to_pert : pert_nodes) {
          auto it = std::find(node_labels.begin(), node_labels.end(), node_to_pert);
          std::size_t node_index;
          if (it != node_labels.end()) {
            node_index = std::distance(node_labels.begin(), it);
            if (world_rank == 0)
              fprintf(stderr, "pert node is at index %lu\n", node_index);
          } else {
            fprintf(stderr, "Node to pert (%s) not found in multiplex\n", node_to_pert.c_str());
            continue;
          }

          // Recalculate transition matrix
          rwr::scale_node_values(intra_tran, N, node_index, 0.0);
        }
        fprintf(stderr, "\tRescaled pert node edges\n");
        // After scaling each edge associated with the pert_nodes,
        // we need to renormalize the intra-layer transition matrix
        // and scale by (1-delta)
        intra_tran.col_normalize();
        fprintf(stderr, "\tRenormalized intra_tran\n");
        intra_tran.scale_values(1.0 - delta);
        fprintf(stderr, "Recalculated transition matrix\n");

        // vector to save encodings for current rank
        std::vector<double> pert_encodings;
        // Transfer transition matrix to GPU
        rwr.transfer_intra_transition_matrix(intra_tran.get_values(),
                                             intra_tran.get_col_idx(),
                                             intra_tran.get_row_ptr());
        rwr.transfer_inter_transition_matrix(inter_tran);

        // Encode seed vectors in a batch or micro-batch
        rwr.rwr_batch(pert_encodings, worker_seeds, node_labels, reduction_method, {}, convergence_threshold);


        std::string encoding_file_name = output_dir + "pert_encodings.tsv";
        std::vector<double> all_pert_encodings;
        parallel::gather(all_pert_encodings, pert_encodings, 0);

        if (world_rank == 0) {
          file_io::print_column_major_matrix(encoding_file_name,
                                             all_pert_encodings,
                                             encoding_size,
                                             all_seeds_flattened.size(),
                                             node_labels,
                                             all_seeds_flattened,
                                             16,
                                             true,
                                             true);
        }

        if (world_rank == 0)
          fprintf(stderr, "\tFinished pert encodings\n");
        // Calculate distance between vector 'i' in base_encodings and vector 'i' pert_encodings
        if (distance_metric.compare("spearman") == 0) {
          // Calculate ranks
          std::vector<double> pert_ranks;
          rank::rank_all_vectors(
            pert_ranks,
            pert_encodings,
            encoding_size,
            n_encodings,
            rank::TieMethod::Average,
            rank::RankProgression::StandardCompetition,
            false
          );

          // Standardize pert_ranks
          normalize::standard::fit_transform(pert_ranks, encoding_size, n_encodings, false);

          correlate::dot::local_corresponding_columns(pert_distance, pert_distance_offest, base_encodings, pert_ranks, encoding_size, n_encodings, -1.0 / (encoding_size - 1), 1.0);

        } else if (distance_metric.compare("pearson") == 0) {
          // Standardize base_encodings
          normalize::standard::fit_transform(pert_encodings, encoding_size, n_encodings, false);

          correlate::dot::local_corresponding_columns(pert_distance, pert_distance_offest, base_encodings, pert_encodings, encoding_size, n_encodings, -1.0 / (encoding_size - 1), 1.0);
          
        } else if (distance_metric.compare("dot") == 0 ) {
          correlate::dot::local_corresponding_columns(pert_distance, pert_distance_offest, base_encodings, pert_encodings, encoding_size, n_encodings, -1.0, 1.0);
        } else {
          correlate::cos::local_corresponding_columns(pert_distance, pert_distance_offest, base_encodings, pert_encodings, encoding_size, n_encodings, -1.0, 1.0);
        }

        pert_distance_offest += n_encodings;
        
      }

        // std::string dist_matrix_file_name = output_dir + distance_metric + "_perturbation_distance_matrix" + std::to_string(world_rank) + ".tsv";
        // file_io::print_column_major_matrix(dist_matrix_file_name,
        //                                  pert_distance,
        //                                  worker_seeds_flattened.size(),
        //                                  pert_node_sets.size(),
        //                                  worker_seeds_flattened,
        //                                  {},
        //                                  16,
        //                                  false);

      // Gather pert_distance values from all workers
      std::vector<double> all_pert_distance;
      parallel::gather_and_row_join_column_major(all_pert_distance, pert_distance, pert_nodes_flattened.size());
      // parallel::gather(all_pert_distance, pert_distance, 0);

      // Record distance matrix
      if (world_rank == 0) {
        std::string dist_matrix_file_name = output_dir + distance_metric + "_perturbation_distance_matrix.tsv";
        file_io::print_column_major_matrix(dist_matrix_file_name,
                                         all_pert_distance,
                                         all_seeds_flattened.size(),
                                         pert_nodes_flattened.size(),
                                         all_seeds_flattened,
                                         pert_nodes_flattened,
                                         16,
                                         false);
      }
    }    
  }
  MPI_Finalize();
  return 0;
}