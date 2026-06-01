#include <string>
#include <CLI/CLI.hpp>
#include <filesystem>

#ifdef USE_OPENMP
#include <omp.h>
#endif

#include <multiplex/Multiplex.hpp>
#include <network/Network.hpp>
#include <file_io/matrix_io.hpp>

namespace fs = std::filesystem;

void check_and_create_path(const fs::path p) {
  // Check if p exists
  if (fs::exists(p)) {
    if (fs::is_regular_file(p)) {
      throw std::invalid_argument("gene_layer_map - " + p.string() + " is a regulare file, not a directory");
    }
    if (!fs::is_directory(p)) {
      throw std::invalid_argument("gene_layer_map - " + p.string() + " is another type of file (e.g. symlink, block device, etc)");
    }
  } else {
    // create directory
    fs::create_directories(p);
  }
}

MergeMethod get_method_from_string(const std::string& s) {
  if (s == "max") {
    return MergeMethod::Max;
  } else if (s == "min") {
    return MergeMethod::Min;
  } else if (s == "all") {
    return MergeMethod::All;
  } else if (s == "sum") {
    return MergeMethod::Sum;
  } else if (s == "mean") {
    return MergeMethod::Mean;
  } else {
    throw std::runtime_error("get_method_from_string - Unknown merge method");
  }
}

int main(int argc, char** argv) {
  CLI::App app{"Create a gene-layer presence map"};

  std::string flist, output_dir, merge_method = "max", runtag;
  bool no_edgelist_headers = false;
  // Path to the file containing paths to and names of each layer in the multiplex
  app.add_option("-f,--flist", flist, "Path to the file containing paths to and names of each layer in the multiplex")
    ->required();
  app.add_flag("--no_edgelist_headers", no_edgelist_headers, "Indicates that the edge lists contains no headers");
  app.add_option("-o,--output_dir", output_dir, "Output directory")
    ->default_val("./");
  app.add_option("--merge_method", merge_method, "Indicates method used to merge edges in multiplex to single layer")
    ->check(CLI::IsMember({"max", "min", "all", "sum", "mean"}))
    ->default_val("max");
  app.add_option("--runtag", runtag, "string to append to output")
    ->required();

  // Parse user inputs and compare against CLI requirements
  try { 
    (app).parse((argc), (argv));
  } catch (const CLI::ParseError &e) {
    int cli_error = (app).exit(e);
    return cli_error;
  }

  std::filesystem::path output_path = output_dir;
  check_and_create_path(output_path);

  #ifdef USE_OPENMP
    fprintf(stderr, "Max OpenMP threads: %d\n", omp_get_max_threads());
  #else
    fprintf(stderr, "Using a single thread\n");
  #endif

  Multiplex mp(flist, !no_edgelist_headers);
  const auto node_labels = mp.get_nodes();

  fprintf(stderr, "Read in multiplex\n");
  fprintf(stderr, "\tmultiplex has %lu layers\n", mp.n_layers());
  fprintf(stderr, "\tmultiplex has %lu nodes\n", mp.n_nodes());
  fprintf(stderr, "\tmultiplex has %lu intra-layer edges.\n", mp.get_n_intra_edges());

  // Network merged = mp.merge_layers(get_method_from_string(merge_method));
  // auto components = merged.connected_components();
  // fprintf(stderr, "The multiplex has %lu components\n", components.size());
  // for (std::size_t i_c = 0; i_c < components.size(); ++i_c) {
  //   fprintf(stderr, "Components %lu has %lu nodes\n", i_c + 1, components[i_c].size());
  // }

  auto layer_names = mp.get_layer_names();
  std::vector<bool> nodes_by_layer = mp.get_nodes_by_layer(node_labels);
  fs::path nodes_by_layer_file_name = output_path / ( runtag + "nodes_by_layer.tsv");
  file_io::print_column_major_matrix(
    nodes_by_layer_file_name.string(),
    nodes_by_layer,
    layer_names.size(),
    node_labels.size(),
    layer_names,
    node_labels,
    true
  );

  fprintf(stderr, "Finished mapping nodes\n");

  // Calculate number of nodes and edges per network
  fs::path network_stats_file_name = output_path / ( runtag + "network_stats.tsv");
  FILE* net_stats = fopen(network_stats_file_name.string().c_str(), "w");
  
  for (std::size_t l = 0; l < mp.n_layers(); ++l) {
    const auto net = mp.get_network(l);

    fprintf(net_stats, "%s\t%lu\t%lu\n", layer_names[l].c_str(), net.get_n_nodes(), net.get_n_edges());
  }

  fclose(net_stats);

  
  return 0;
}