#include "file_io/binary_io.hpp"
#include <filesystem>
#include <regex>

namespace file_io {

std::vector<std::pair<int, int>> find_block_files(const std::string& directory, const std::string& file_prefix) {
  std::vector<std::pair<int, int>> blocks;
  std::string pattern_str = file_prefix + R"(_(\d+)_(\d+)\.bin)";
  std::regex pattern(pattern_str);

  for (const auto& entry : std::filesystem::directory_iterator(directory)) {
    const std::string filename = entry.path().filename().string();

    std::smatch match;
    if (std::regex_match(filename, match, pattern)) {
      int i = std::stoi(match[1]);
      int j = std::stoi(match[2]);
      blocks.emplace_back(i,j);
    }
  }

  return blocks;
}
  
} // namespace file_io
