#include <utils/vector_utils.hpp>

namespace utils {

std::string concate(const std::vector<std::string>& __x, const std::string& sep) {
  if (__x.empty()) {
    return "";
  }

  std::string output = __x[0];

  for (std::size_t i = 1; i < __x.size(); ++i) {
    output = output + sep + __x[i];
  }

  return output;
}

} // namespace utils