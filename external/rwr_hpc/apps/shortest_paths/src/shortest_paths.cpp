#include <string>

#include <CLI/CLI.hpp>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <filesystem>

#include <timer/Timer.hpp>
#include <multiplex/Multiplex.hpp>
#include <network/Network.hpp>
#include <file_io/matrix_io.hpp>

namespace fs = std::filesystem;

void check_and_create_path(const fs::path p) {
  // Check if p exists
  if (fs::exists(p)) {
    if (fs::is_regular_file(p)) {
      throw std::invalid_argument("shortest_paths - " + p.string() + " is a regulare file, not a directory");
    }
    if (!fs::is_directory(p)) {
      throw std::invalid_argument("shortest_paths - " + p.string() + " is another type of file (e.g. symlink, block device, etc)");
    }
  } else {
    // create directory
    fs::create_directories(p);
  }
}

static inline std::string trim(const std::string& s) {
  size_t start = s.find_first_not_of(" \t\r\n");
  if (start == std::string::npos) return "";
  size_t end = s.find_last_not_of(" \t\r\n");
  return s.substr(start, end - start + 1);
}

void read_seeds(std::vector<std::string>& seeds,
  const std::string& file_name,
  const bool skip_first_col,
  const char delim)
{
  std::ifstream infile(file_name);
  if (!infile.is_open()) {
    throw std::runtime_error("read_seeds: Unable to open file '" + file_name + "'");
  }

  std::string line;
  size_t line_number = 0;

  while (std::getline(infile, line)) {
    line_number++;
    line = trim(line);

    // Skip empty lines
    if (line.empty()) continue;

    std::stringstream ss(line);
    std::string col;

    std::vector<std::string> cols;
    while (std::getline(ss, col, delim)) {
      cols.push_back(trim(col));
    }

    // Required column index
    size_t index = skip_first_col ? 1 : 0;

    if (cols.size() <= index) {
      std::stringstream err;
      err << "read_seeds: Line " << line_number
          << " does not contain required column index " << index
          << ". Full line: '" << line << "'";
      throw std::runtime_error(err.str());
    }

    // Add the seed
    seeds.push_back(cols[index]);
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

std::vector<std::string> read_file_by_line(const std::string& file_name) {
  std::ifstream infile(file_name);
  if (!infile.is_open()) {
    throw std::runtime_error("read_file_by_line: Unable to open file '" + file_name + "'");
  }

  std::vector<std::string> output;

  std::string line;
  while (std::getline(infile, line)) {
    line = trim(line);

    // Skip empty lines
    if (line.empty()) continue;

    // Add the line
    output.push_back(line);
  }

  return output;
}

int main(int argc, char** argv) {

  Timer timer;
  timer.start();

  CLI::App app{"Shortest paths"};

  // CLI arguments
  std::string flist, sources_file_raw, targets_file_raw = "", output_dir_raw = ".", runtag_raw = "", merge_method = "max";
  bool no_edgelist_headers = false, no_set_ids = false, ignore_weights = false, multifile = false;

  app.add_option("-f,--flist", flist, "Path to the file containing paths to and names of each layer in the multiplex")
    ->required();
  app.add_flag("--no_edgelist_headers", no_edgelist_headers, "Indicates that the edge lists contains no headers");
  app.add_option("-s,--sources_file", sources_file_raw, "Tab delimited file containing nodes that act as shortest paths sources.\n")
    ->required();
  app.add_option("-t,--targets_file", targets_file_raw, "Tab delimited file containing nodes that act as shortest paths targets.\n"
                                                    "If no targets file is provided sources will act as targets\n")
    ->default_val("");
  app.add_flag("--no_set_ids", no_set_ids, "Indicates that the sources and targets files contains no set ids\n."
                                           "These ids are stored as the first value in each row.");
  app.add_option("--merge_method", merge_method, "Indicates method used to merge edges in multiplex to single layer")
    ->check(CLI::IsMember({"max", "min", "all", "sum", "mean"}))
    ->default_val("max");
  app.add_option("-o,--output_dir", output_dir_raw, "Output directory")
    ->default_val("./");
  app.add_option("--runtag", runtag_raw, "Name pre-pended to output_files specifiying run.")
    ->required();
  app.add_flag("--ignore_weights", ignore_weights, "Ignore weights when finding the shortest paths");
  app.add_flag("--multifile", multifile, "Indicates that multpile node sets will be run"
                                         "Expects: `sources_file` to point to a file, where each line contains a file to a set of source nodes"
                                         "       : `targets_file` is empty or points to a file, where each line contains a file to a set of target nodes"
                                         "       : `run_tag` to point to files where each line contains a run tag"
                                         "       : `output_dir` to point to files where each line contains a output_dir");

  // Parse user inputs and compare against CLI requirements
  try { 
    (app).parse((argc), (argv));
  } catch (const CLI::ParseError &e) {
    int cli_error = (app).exit(e);
    return cli_error;
  }


  std::vector<std::string> sources_files, targets_files, run_tags, output_dirs;

  if (multifile) {
    sources_files = read_file_by_line(sources_file_raw);
    if (!targets_file_raw.empty()) {
      targets_files = read_file_by_line(targets_file_raw);
    }
    run_tags = read_file_by_line(runtag_raw);
    if (output_dir_raw != "./") {
      output_dirs = read_file_by_line(output_dir_raw);
    } else {
      for (std::size_t i = 0; i < targets_files.size(); ++i) {
        output_dirs.push_back(output_dir_raw);
      }
    }

    if (sources_files.size() != run_tags.size()) {
      throw std::invalid_argument("Shortest_paths - size mismatch between sources_files and run_tags using multifile mode");
    }
    if (sources_files.size() != output_dirs.size()) {
      throw std::invalid_argument("Shortest_paths - size mismatch between sources_files and output_dirs using multifile mode");
    }
    if (!targets_files.empty() && sources_files.size() != targets_files.size()) {
      throw std::invalid_argument("Shortest_paths - size mismatch between sources_files and targets_files using multifile mode");
    }
  } else {
    sources_files.push_back(sources_file_raw);
    if (!targets_file_raw.empty()) targets_files.push_back(targets_file_raw);
    run_tags.push_back(runtag_raw);
    output_dirs.push_back(output_dir_raw);
  }

  // Check if output_dir is not a file and create the directory if needed
  for (const auto& output_dir : output_dirs) {
    fs::path output_path = output_dir;
    check_and_create_path(output_path);
  }
  

  // Load multiplex
  timer.restart();
  Multiplex mp = Multiplex(flist, !no_edgelist_headers);
  fprintf(stderr, "Took %lf seconds to read multiplex\n", timer.elapsed_wall_time());
  fprintf(stderr, "Multiplex has %lu nodes, %lu layers, and %lu edges\n", mp.n_nodes(), mp.n_layers(), mp.get_n_intra_edges());
  timer.restart();
  const auto node_labels = mp.get_nodes();

  // Merge all layers into single network
  Network merged = mp.merge_layers(get_method_from_string(merge_method));
  fprintf(stderr, "Took %lf seconds to merge multiplex layers\n", timer.elapsed_wall_time());
  fprintf(stderr, "Merged network has %lu node and %lu edges\n", merged.get_n_nodes(), merged.get_n_edges());
  timer.restart();

  // Convert edge weights to distance for shortest paths calculations
  merged.convert_edges_to_distance();
  fprintf(stderr, "Took %lf seconds to convert edge weights to distances\n", timer.elapsed_wall_time());
  timer.restart();

  // /* 
  // // Determine number of components (connected subgraphs) in the multiplex
  // */
  // auto components = merged.connected_components();
  // fprintf(stderr, "The multiplex has %lu components\n", components.size());
  // for (std::size_t i_c = 0; i_c < components.size(); ++i_c) {
  //   fprintf(stderr, "Components %lu has %lu nodes\n", i_c + 1, components[i_c].size());
  // }
  // auto layer_contribution_per_component = mp.get_layer_contribution_per_component(components);
  // auto layers = mp.get_layer_names();
  // std::vector<std::string> comp_names;
  // for (std::size_t i = 0; i < components.size(); ++i) {
  //   comp_names.push_back("comp_" + std::to_string(i));
  // }
  // // Build final file path (automatic slash handling)
  // std::filesystem::path layer_contribution_per_component_file = output_path / ( runtag + "layer_contribution_per_component.tsv");
  // std::string layer_contribution_per_component_file_str = layer_contribution_per_component_file.string();
  // file_io::print_column_major_matrix(
  //   layer_contribution_per_component_file_str,
  //   layer_contribution_per_component,
  //   layers.size(),
  //   comp_names.size(),
  //   layers,
  //   comp_names
  // );

  // --- Loop through all files (if multifile is false, this will be a single loop) ---
  for (std::size_t i_file = 0; i_file < sources_files.size(); ++i_file) {
    std::string sources_file = sources_files[i_file];
    std::string targets_file;
    if (!targets_files.empty()) targets_file = targets_files[i_file];

    std::string output_dir = output_dirs[i_file];
    std::string runtag = run_tags[i_file];
    fs::path output_path = output_dir;

    // Load source nodes
    std::unordered_set<std::string> sources, targets, sources_in_mp;

    std::vector<std::string> sources_raw;
    read_seeds(sources_raw, sources_file, !no_set_ids, '\t');
    sources.insert(sources_raw.begin(), sources_raw.end());
    fprintf(stderr, "Took %lf seconds to read %lu nodes\n", timer.elapsed_wall_time(), sources_raw.size());
    timer.restart();

    // Check if any seeds are not in multiplex
    std::unordered_set<std::string> mp_nodes_set(node_labels.begin(), node_labels.end());
    for (auto s : sources) {
      if (mp_nodes_set.find(s) == mp_nodes_set.end()) {
        fprintf(stderr, "Source label %s is not in multiplex\n", s.c_str());
      } else {
        sources_in_mp.insert(s);
      }
    }

    if (targets.empty()) {
      targets = sources_in_mp;
    }

    // Calculate all shortest paths between each s \in S and each t \in T (except s == t)
    auto all_shortest_paths = merged.find_all_shortest_paths(sources_in_mp, targets, !ignore_weights);
    fprintf(stderr, "Took %lf seconds to calculate all shortest paths (%lu)\n", timer.elapsed_wall_time(), all_shortest_paths.size());
    timer.restart();

    // Determine which layer each shortest path edge is located. Save states and print results to file
    auto path_to_string = [](const std::vector<std::string>& path) {
      std::ostringstream oss;
      for (size_t i = 0; i < path.size(); ++i) {
        if (i > 0) oss << "->";
        oss << path[i];
      }
      return oss.str();
    };

    // Build final file path (automatic slash handling)
    std::filesystem::path out_file = output_path / ( runtag + "_shortest_paths.tsv");
    // Convert to native string if needed (UTF-8 on Linux/Mac, wide on Windows)
    std::string out_file_str = out_file.string();

    FILE* f = fopen(out_file_str.c_str(), "w");
    if (!f) {
      std::cerr << "Error: could not open output file: " << out_file_str << "\n";
      return 1;
    }

    fprintf(f, "from\tto\tweight\ttype\tpathname\tpathlength\tpathelements\n");

    std::unordered_map<std::string, uint64_t> layer_counts;

    std::string out, line;
    out.reserve(64 * 1024);
    line.reserve(4096);
    char buf[64];

    for (const auto& path : all_shortest_paths) {
      if (path.empty()) {
        fprintf(stderr, "Empty path\n");
        continue;
      }

      const std::string path_name = path[0] + "_" + path.back();
      const std::string path_elements = path_to_string(path);
      const std::size_t path_length = path.size() - 1;

      out.clear();

      for (std::size_t i = 0; i < path.size() - 1; ++i) {
        const std::string& src = path[i];
        const std::string& tgt = path[i+1];

        auto edges = mp.get_layers_with_highest_weight(src, tgt);

        if (edges.empty()) {
          fprintf(stderr, "Edge not found in multiplex (%s, %s)\n", src.c_str(), tgt.c_str());
          exit(EXIT_FAILURE);
        }

        for (const auto& edge : edges) {
          layer_counts[edge.first]++;

          snprintf(buf, sizeof(buf), "%.17e", edge.second);  // scientific notation
          line.clear();
          line.append(src);
          line.push_back('\t');
          line.append(tgt);
          line.push_back('\t');
          line.append(buf);
          line.push_back('\t');
          line.append(edge.first);
          line.push_back('\t');
          line.append(path_name);
          line.push_back('\t');
          line.append(std::to_string(path_length));
          line.push_back('\t');
          line.append(path_elements);
          line.push_back('\n');

          out.append(line);
        }
      }

      fwrite(out.data(), 1, out.size(), f);
    }

    fclose(f);

    // Sort layer count in decreasing order than print
    std::vector<std::pair<std::string, uint64_t>> layer_vec(layer_counts.begin(), layer_counts.end());

    std::sort(
      layer_vec.begin(),
      layer_vec.end(),
      [](const auto& a, const auto& b) {
        return a.second > b.second;  // largest count first
      }
    );

    std::ofstream summary_file(output_path / (runtag + "_layer_counts.tsv"));
    for (const auto& [layer_name, count] : layer_vec) {
      summary_file << layer_name << "\t" << count << "\n";
    }
  }
}