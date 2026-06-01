#include <CLI/CLI.hpp>
#include <string>

#include "../include/rwr_loe.hpp"

#include <multiplex/Multiplex.hpp>
#include <sparse/CSR_Matrix.hpp>
#include <rwr/rwr.hpp>
#include <rank/rank.hpp>
#include <timer/Timer.hpp>
#include <sort/sort.hpp>
#include <utils/vector_utils.hpp>

int main(int argc, char* argv[]) {

  Timer timer;
  timer.start();

  CLI::App app{"Lines-of-evidence using RWR embeddings on multiplex networks"};

  // --- CLI arguments ---
  std::string flist, seed_file, query_file, reduction_method, output_dir, run_tag;
  bool no_edgelist_headers = false, no_set_ids = false, verbose = false;
  double r, delta, convergence_threshold, numranked;

  app.add_option("-f,--flist", flist, "Path to the file containing paths to and names of each layer in the multiplex")
    ->required();
  app.add_flag("--no_edgelist_headers", no_edgelist_headers, "Indicates that the edge lists contains no headers");
  app.add_option("-s,--seed_file", seed_file, "Tab delimited file containing seeds to initalize RWR probabilty vectors with.\n"
                                              "Seeds on a single line will be combined in a single vector.\n"
                                              "Seeds on mulitple lines will be encoded in seperate vectors.\n"
                                              "Any empty value will treat each node in the multiplex as a seed separately.")
    ->required();
  app.add_option("-q,--query_file", query_file, "Tab delimited file containing seeds to initalize RWR probabilty vectors with.\n"
                                              "Seeds on a single line will be combined in a single vector.\n"
                                              "Seeds on mulitple lines will be encoded in seperate vectors.\n"
                                              "Any empty value will treat each node in the multiplex as a seed separately.")
    ->default_val("");
  app.add_flag("--no_set_ids", no_set_ids, "Indicates that the pertubation_file and seed_file contains no set ids\n.");
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
  app.add_option("--numranked", numranked, "roprtion of ranked genes to return. (e.g. 0.1 will return the top 10%.)")
    ->default_val(1.0)
    ->check(CLI::Range(0.0, 1.0));
  app.add_flag("--verbose", verbose, "Indicates that verbose output should be reported");
  app.add_option("--run_tag", run_tag, "Name used to alias this run")
    ->default_val("");

  // --- Parse user inputs and compare against CLI requirements ---
  try {
    (app).parse((argc), (argv));
  } catch (const CLI::ParseError &e) {
    int cli_error = (app).exit(e);
    return cli_error;
  }

  // --- Check if output_dir exist ---
  std::filesystem::path output_path = output_dir;
  if (!std::filesystem::is_directory(output_path)) {
    std::filesystem::create_directories(output_path);
  }
  
  if (verbose) {
    timer.restart();
  }
  // --- Read in multiplex and save local variables ---
  Multiplex mp(flist, !no_edgelist_headers);
  auto node_labels = mp.get_nodes();
  const std::size_t N = mp.n_nodes();
  const std::size_t L = mp.n_layers();
  if (verbose) {
    fprintf(stderr, "Took %lf seconds to read in multiplex and save local vars\n", timer.elapsed_wall_time());
    fprintf(stderr, "Multiplex has %lu unique nodes\n", node_labels.size());
    timer.restart();
  }

  // --- Read in seed file and query file (if provided)
  std::vector<std::vector<std::string>> seed_set, query_set;
  rwr::read_seeds(seed_set, seed_file, !no_set_ids);
  rwr::remove_seeds_not_in_list(seed_set, node_labels);
  rwr::remove_duplicate_seed_vectors(seed_set);

  // --- Combine all seeds into a single vector.
  auto flattened_seeds = rwr::flatten(seed_set);
  seed_set.clear();
  seed_set.push_back(flattened_seeds);

  if (!query_file.empty()) {
    rwr::read_seeds(query_set, query_file, !no_set_ids);
    rwr::remove_seeds_not_in_list(query_set, node_labels);
    rwr::remove_duplicate_seed_vectors(query_set);
  }
  if (verbose) {
    fprintf(stderr, "Took %lf seconds to read in seeds\n", timer.elapsed_wall_time());
    timer.restart();
  }


  // --- Create intra-layer and inter-layer transition matrix parts ---
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

  std::size_t encoding_size = rwr::get_encoding_size(reduction_method, N, L);
  std::size_t n_encodings = seed_set.size();
  std::vector<double> encoding;

  // Encapsulte RWR Worker for auto deletion after use
  {
    // --- Initialize RWR worker and transfer transition matrix to GPU ---
    rwr::RWR_Worker rwr(N, L, intra_tran.nnz(), n_encodings, r);
    rwr.transfer_intra_transition_matrix(intra_tran.get_values(),
                                         intra_tran.get_col_idx(),
                                         intra_tran.get_row_ptr());
    rwr.transfer_inter_transition_matrix(inter_tran);

    // --- Run RWR using all seeds ---
    rwr.rwr_batch(encoding, seed_set, node_labels, reduction_method, {}, convergence_threshold);
  }

  // --- Rank encoding. Use tie_method == "min" ---
  std::vector<double> ranks;
  rank::rank_all_vectors(
    ranks,
    encoding,
    encoding_size,
    n_encodings,
    rank::TieMethod::Min,
    rank::RankProgression::StandardCompetition,
    false
  );

  // --- Create container for labels, encoding, and ranks ---
  std::vector<Row> data;
  create_row_vector(data, node_labels, encoding, ranks);

  // --- Remove nodes from seed set ---
  // Create set of seeds
  auto seeds = seed_set[0];
  remove_ids(data, seeds);

  // --- If a query set was provided, filter results to just those entries ---
  if (!query_set.empty()) {
    auto query = rwr::flatten(query_set);
    keep_ids(data, query);
  }

  // --- Sort data based on rank ---
  sort_by_rank(data);

  // --- Determine number of entries to print ---
  const std::size_t num_entries = static_cast<std::size_t>(data.size() * numranked);

  // --- Write results to file ---
  // Build final file path
  std::filesystem::path out_file = output_path / ( "RWR-LOE_" + run_tag + ".ranks.tsv");
  std::string out_file_str = out_file.string();

  // Write to file
  auto layers = mp.get_layer_names();
  std::string networks = utils::concate(layers);

  std::filesystem::path input_path = flist;
  std::string base_name = input_path.stem().string();

  if (run_tag.empty()) {
    run_tag = base_name;
  }

  write_table(
    out_file_str,
    data,
    num_entries,
    flattened_seeds.size(),
    flattened_seeds.size(),
    networks,
    run_tag,
    base_name
  );

  return 0;
}

void create_row_vector(
  std::vector<Row>& data,
  const std::vector<std::string>& ids,
  const std::vector<double> scores,
  const std::vector<double> ranks
) {
  if (ids.size() != scores.size()) {
    throw std::invalid_argument(
      "rwr_low::create_row_vector - ids and scores must be the same size. "
      "ids.size() = " + std::to_string(ids.size()) +
      ", scores.size() = " + std::to_string(scores.size())
    );
  }
  if (ids.size() != ranks.size()) {
    throw std::invalid_argument(
      "rwr_low::create_row_vector - ids and ranks must be the same size. "
      "ids.size() = " + std::to_string(ids.size()) +
      ", ranks.size() = " + std::to_string(ranks.size())
    );
  }

  data.clear();
  data.reserve(ids.size());

  for (std::size_t i = 0; i < ids.size(); ++i) {
    data.push_back({ids[i], scores[i], ranks[i]});
  }
}

void remove_ids(
  std::vector<Row>& data,
  const std::vector<std::string>& ids_to_remove
) {
  std::unordered_set<std::string> id_set(ids_to_remove.begin(), ids_to_remove.end());
  // Remove row of `r.id` is found in `id_set`
  data.erase(
    std::remove_if(
      data.begin(),
      data.end(),
      [&](const Row& r) {
        return id_set.find(r.id) != id_set.end(); // Found in set
      }),
    data.end()
  );
}

void keep_ids(
  std::vector<Row>& data,
  const std::vector<std::string>& ids_to_keep
) {
  std::unordered_set<std::string> id_set(ids_to_keep.begin(), ids_to_keep.end());
  // Remove row of `r.id` is NOT found in `id_set`
  data.erase(
    std::remove_if(
      data.begin(),
      data.end(),
      [&](const Row& r) {
        return id_set.find(r.id) == id_set.end(); // Not found in set
      }),
    data.end()
  );
}

void sort_by_rank(std::vector<Row>& data) {
 std::sort(
    data.begin(),
    data.end(),
    [](const Row& a, const Row& b) {
      return std::tie(a.rank, a.id) < std::tie(b.rank, b.id);
    }
  );
}

void write_table(
  const std::string& filename,
  const std::vector<Row>& data,
  const std::size_t num_entries,
  const std::size_t num_in_network,
  const std::size_t num_seeds,
  const std::string& networks,
  const std::string& runtag,
  const std::string& seed_geneset
) {
  // --- Check inputs ---
  if (num_entries > data.size()) {
    throw std::invalid_argument(
      "rwr_low::write_table - num_entries must be less than or equal to the number of data rows. "
      "data.size() = " + std::to_string(data.size()) +
      ", num_entries = " + std::to_string(num_entries)
    );
  }

  std::ofstream out(filename);
  if (!out) { 
    throw std::runtime_error("rwr_low::write_table - failed to open file: " + filename);
  }

  // Optional header
  out << "NodeNames\tScores\trank\tnum_in_network\tnum_seeds\tnetworks\tmodname\tseed_geneset\n";

  out << std::setprecision(15);
  out << std::scientific;

  for (std::size_t i = 0; i < num_entries; ++i) {
    out << data[i].id << '\t'
        << data[i].score << '\t'
        << static_cast<uint64_t>(std::trunc(data[i].rank)) << '\t'
        << num_in_network << '\t'
        << num_seeds << '\t'
        << networks << '\t'
        << runtag << '\t'
        << seed_geneset << '\n';
  }
}
