#include <string>
#include <stdexcept>

#include <CLI/CLI.hpp>
#include <multiplex/Multiplex.hpp>
#include <network/Network.hpp>

namespace fs = std::filesystem;

void check_and_create_path(const fs::path p);

int main(int argc, char* argv[]) {
  // --- CLI arguments ---
  CLI::App app{"Create seed sets for each component of multiplex"};

  std::string flist, output_dir, runtag;
  bool no_edgelist_headers = false;

  app.add_option("-f,--flist", flist, "Path to the file containing paths to and names of each layer in the multiplex")
      ->required();
  app.add_flag("--no_edgelist_headers", no_edgelist_headers, "Indicates that the edge lists contains no headers");
  app.add_option("--runtag", runtag, "string to append to output")
    ->required();
  app.add_option("-o,--output_dir", output_dir, "Output directory")
    ->default_val("./");

  // --- Parse user inputs and compare against CLI requirements ---
  try {
    (app).parse((argc), (argv));
  } catch (const CLI::ParseError &e) {
    int cli_error = (app).exit(e);
    return cli_error;
  }

  // -- Check that out_path is valid and create the directory if needed ---
  fs::path output_path = output_dir;
  check_and_create_path(output_path);

  // --- Read in multiplex ---
  Multiplex mp(flist, !no_edgelist_headers);

  // --- Merge all layers to single network for BFS based connected component identification ---
  Network merged = mp.merge_layers(MergeMethod::All);
  auto components = merged.connected_components();
  fprintf(stderr, "The multiplex has %lu components\n", components.size());
  for (std::size_t i_c = 0; i_c < components.size(); ++i_c) {
    fprintf(stderr, "Component %lu has %lu nodes\n", i_c + 1, components[i_c].size());
  }

  for (std::size_t i_c = 0; i_c < components.size(); ++i_c) {
    std::string set_id = runtag + "_comp" + std::to_string(i_c + 1);

    std::string name = "_comp" + std::to_string(i_c + 1) + "_seeds.tsv";
    fs::path comp_file = output_path / ( runtag + name);

    name = comp_file.c_str();

    FILE* fp = nullptr;
    fp = fopen(name.c_str(), "w");

    for (auto& label : components[i_c]) {
      fprintf(fp, "%s\t%s\n", set_id.c_str(), label.c_str());
    }
    
    fclose(fp);
  }
  

  return 0;
}

void check_and_create_path(const fs::path p) {
  // Check if p exists
  if (fs::exists(p)) {
    if (fs::is_regular_file(p)) {
      throw std::invalid_argument("tree_walker - " + p.string() + " is a regulare file, not a directory");
    }
    if (!fs::is_directory(p)) {
      throw std::invalid_argument("tree_walker - " + p.string() + " is another type of file (e.g. symlink, block device, etc)");
    }
  } else {
    // create directory
    fs::create_directories(p);
  }
}