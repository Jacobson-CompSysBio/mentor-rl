#pragma once

#include <string>
#include <vector>

struct Row {
  std::string id;
  double score;
  double rank;
};

void create_row_vector(
  std::vector<Row>& data,
  const std::vector<std::string>& ids,
  const std::vector<double> scores,
  const std::vector<double> ranks
);

void remove_ids(
  std::vector<Row>& data,
  const std::vector<std::string>& ids_to_remove
);

void keep_ids(
  std::vector<Row>& data,
  const std::vector<std::string>& ids_to_keep
);

void sort_by_rank(std::vector<Row>& data);

void write_table(
  const std::string& filename,
  const std::vector<Row>& data,
  const std::size_t num_entries,
  const std::size_t num_in_network,
  const std::size_t num_seeds,
  const std::string& networks,
  const std::string& runtag,
  const std::string& seed_geneset
);
