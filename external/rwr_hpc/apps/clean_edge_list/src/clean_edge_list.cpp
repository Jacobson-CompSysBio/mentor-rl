#include <CLI/CLI.hpp>
#include <string>
#include <unordered_map>
#include <vector>
#include <filesystem>

struct Edge {
  uint32_t target = 0;
  double weight = 0.0;
  std::string meta;
};

enum class DuplicateMethod {
  Max,
  Min,
  Sum,
  Mean
};

DuplicateMethod convert_str_to_enum(const std::string& s) {
  if (s == "max") {
    return DuplicateMethod::Max;
  } else if (s == "min") {
    return DuplicateMethod::Min;
  } else if (s == "sum") {
    return DuplicateMethod::Sum;
  } else if (s == "mean") {
    return DuplicateMethod::Mean;
  }
}

int main(int argc, char* argv[]) {

  CLI::App app{"Cleans an edge list of self-loops, duplicate edges, and directed edges"};

  std::string edge_list_file, output_dir, duplicate_edge_method;
  bool no_edgelist_headers = false, directed = false, allow_self_loops = false;
  char sep = '\t';

  // --- CLI arguments ---
  app.add_option("--edge_list", edge_list_file, "Path to the edge list file")
    ->required();
  app.add_flag("--no_edgelist_headers", no_edgelist_headers, "Indicates that the edge list contains no headers");
  app.add_flag("--directed", directed, "Indicates that the network should be treated as directed");
  app.add_option("-o,--output_dir", output_dir, "Output directory")
    ->default_val("./");
  app.add_option("--duplicate_edge_method", duplicate_edge_method, "The method used to reduce multiple edges with same source and target")
    ->check(CLI::IsMember({"max", "min", "sum", "mean"}))
    ->default_val("max");
  app.add_flag("--allow_self_loops", allow_self_loops, "Indicates that self-loops are allowed");
  
  // --- Parse user inputs and compare against CLI requirements ---
  try {
    (app).parse((argc), (argv));
  } catch (const CLI::ParseError &e) {
    int cli_error = (app).exit(e);
    return cli_error;
  }

  DuplicateMethod dup_method = convert_str_to_enum(duplicate_edge_method);

  std::unordered_map<std::string, uint32_t> labels; // lookup: label -> index
  std::vector<std::string> index_to_label; // reverse lookup: index -> label
  std::unordered_map<uint64_t, double> edge_weight;
  std::unordered_map<uint64_t, uint32_t> edge_count;

  fprintf(stderr, "Summary of network: %s\n", edge_list_file.c_str());
  
  // Try to open file
  std::string line, src, tgt, weight_str;
  std::ifstream input(edge_list_file);
  if (!input) {
    throw std::runtime_error("clean_edge_list - input file (" + edge_list_file + ") could not be opened");
  }

  // --- Read file in two passes. On the first pass, identify the unique node values ---
  // Read in headers if indicated
  if (!no_edgelist_headers && !std::getline(input, line)) {
    throw std::runtime_error("clean_edge_list - File missing expected header line");
  }

  while (std::getline(input, line)) {
    if (line.empty()) continue;

    std::istringstream iss(line);

    std::getline(iss, src, sep);
    std::getline(iss, tgt, sep);
    index_to_label.push_back(src);
    index_to_label.push_back(tgt);
  }

  std::sort(index_to_label.begin(), index_to_label.end());
  index_to_label.erase(std::unique(index_to_label.begin(), index_to_label.end()), index_to_label.end());

  const std::size_t N = index_to_label.size();
  fprintf(stderr, "%lu unique nodes in network\n", N);

  labels.reserve(N);
  uint32_t next_label = 0;
  for (const auto& label : index_to_label) {
    labels.emplace(label, next_label++);
  }

  // --- Rewind for second pass ---
  input.clear();                  // Clear EOF and any error flags
  input.seekg(0, std::ios::beg);  // Move read position to the beginning

  // --- Second pass on file for edges ---
  // Read in headers if indicated
  if (!no_edgelist_headers && !std::getline(input, line)) {
    throw std::runtime_error("clean_edge_list - File missing expected header line");
  }
  std::size_t num_dups_found = 0;
  while (std::getline(input, line)) {
    if (line.empty()) continue;

    std::istringstream iss(line);

    std::getline(iss, src, sep);
    std::getline(iss, tgt, sep);

    double weight = 1.0;
    if (iss.peek() != EOF) {
      std::getline(iss, weight_str, sep);

      if (!weight_str.empty()) {
        // Try to convert weight_str to double. throw if conversion fails
        try {
          weight = std::stod(weight_str);
        } catch (std::runtime_error&) {
          throw std::runtime_error("Invalid weight value un edge list: " + weight_str);
        }
      }
    }

    uint32_t src_idx = labels[src];
    uint32_t tgt_idx = labels[tgt];

    if (src_idx == tgt_idx && !allow_self_loops) {
      fprintf(stderr, "Found self-loop with node %s\n", src.c_str());
      continue;
    }

    // If network in undirected, order the source and target
    if (!directed && src_idx > tgt_idx) {
      uint32_t tmp = src_idx;
      src_idx = tgt_idx;
      tgt_idx = tmp;
    }

    // Pack edge
    uint64_t edge_key = (uint64_t(src_idx) << 32 | uint64_t(tgt_idx));

    auto it = edge_weight.find(edge_key);
    if (it == edge_weight.end()) {
      edge_weight[edge_key] = weight;
      if (dup_method == DuplicateMethod::Mean) {
        edge_count[edge_key] = 1;
      }
    } else {
      ++num_dups_found;
      switch (dup_method) {
        case DuplicateMethod::Max:  it->second = std::max(it->second, weight); break;
        case DuplicateMethod::Min:  it->second = std::min(it->second, weight); break;
        case DuplicateMethod::Sum:  it->second += weight; break;
        case DuplicateMethod::Mean: it->second += weight; edge_count[edge_key] += 1; break;
        default: break;
      }
    }
  }

  if (dup_method == DuplicateMethod::Mean) {
    for (auto& kv : edge_weight) {
      kv.second /= double(edge_count[kv.first]);
    }
  }

  // --- Sort Keys ---
  std::vector<uint64_t> keys;
  keys.reserve(edge_weight.size());
  for (const auto& [key, weight] : edge_weight) {
    keys.push_back(key);
  }
  std::sort(keys.begin(), keys.end());

  fprintf(stderr, "Removed %lu duplicate edges\n", num_dups_found);
  fprintf(stderr, "Kept %lu edges\n", edge_weight.size());


  // Check if output_dir
  std::filesystem::path output_path = output_dir;
  if (!std::filesystem::is_directory(output_path)) {
    std::filesystem::create_directories(output_path);
  } else if (!std::filesystem::is_directory(output_dir)) {
    std::cerr << "Error: " << output_dir << " exists but is not a directory.\n";
    return 1;
  }

  // Get file name without extension
  std::filesystem::path input_path = edge_list_file;
  std::string base_name = input_path.stem().string();

  // Create new file name
  std::filesystem::path out_file = output_path / ( base_name + "_cleaned.tsv") ;
  std::string out_file_str = out_file.string();

  FILE* f = fopen(out_file_str.c_str(), "w");
  if (!f) {
    std::cerr << "Error: could not open output file: " << out_file_str << "\n";
    return 1;
  }

  fprintf(f, "source\ttarget\tweight\n");

  // --- Print edges ---
  for (uint64_t key : keys) {
    uint32_t src_idx = key >> 32;
    uint32_t tgt_idx = uint32_t(key & 0xffffffffu);

    fprintf(
      f,
      "%s\t%s\t%.15e\n",
      index_to_label[src_idx].c_str(),
      index_to_label[tgt_idx].c_str(),
      edge_weight[key]
    );
  }

  return 0;
}