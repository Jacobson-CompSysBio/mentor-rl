#include <gtest/gtest.h>
#include <gmock/gmock-matchers.h>
#include <gmock/gmock.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <unordered_set>

#include <rwr/utils.hpp>

namespace rwr {

// Helper to create temporary test file
void write_test_file(const std::string& filename, const std::string& content) {
  std::ofstream out(filename);
  out << content;
  out.close();
}

TEST(TestRwrUtils, ReadSeedsThrowsOnBadFileName) {
  const std::string filename = "bad_seed_file.txt";
  std::vector<std::vector<std::string>> seeds;

  ASSERT_THAT(
    [&](){rwr::read_seeds(seeds, filename); },
    testing::ThrowsMessage<std::runtime_error>("rwr::read_seeds - could not open file: bad_seed_file.txt")
  );
}

TEST(TestRwrUtils, ReadSeedsThrowsOnTooFewColumns) {
  const std::string filename = "seedfile.txt";
  write_test_file(filename,
    "test_set\n"
    "test_set\n"
    "test_set\n"
  );

  std::vector<std::vector<std::string>> seeds;

  ASSERT_THAT(
    [&](){rwr::read_seeds(seeds, filename, true); },
    testing::ThrowsMessage<std::runtime_error>("rwr::read_seeds - Error on line 1: not enough columns to skip first")
  );

  std::remove(filename.c_str());
}

TEST(TestRwrUtils, ReadSeedsSingleSeedPerLineDoNotSkipFirstColumn) {
  const std::string filename = "seedfile.txt";
  write_test_file(filename,
    "gene1\n"
    "gene2\n"
    "gene3\n"
    "gene4\n"
    "gene5\n"
  );

  std::vector<std::vector<std::string>> seeds;

  rwr::read_seeds(seeds, filename, false);

  std::remove(filename.c_str());

  ASSERT_EQ(seeds.size(), 5);
  std::vector<std::string> expected_seed = {"gene1"};
  EXPECT_EQ(seeds[0], expected_seed);

  expected_seed = {"gene2"};
  EXPECT_EQ(seeds[1], expected_seed);

  expected_seed = {"gene3"};
  EXPECT_EQ(seeds[2], expected_seed);

  expected_seed = {"gene4"};
  EXPECT_EQ(seeds[3], expected_seed);

  expected_seed = {"gene5"};
  EXPECT_EQ(seeds[4], expected_seed);
}

TEST(TestRwrUtils, ReadSeedsSingleSeedPerLineSkipFirstColumn) {
  const std::string filename = "seedfile.txt";
  write_test_file(filename,
    "set1\tgene1\n"
    "set1\tgene2\n"
    "set1\tgene3\n"
    "set2\tgene4\n"
    "set2\tgene5\n"
  );

  std::vector<std::vector<std::string>> seeds;

  rwr::read_seeds(seeds, filename, true);

  std::remove(filename.c_str());

  ASSERT_EQ(seeds.size(), 5);
  std::vector<std::string> expected_seed = {"gene1"};
  EXPECT_EQ(seeds[0], expected_seed);

  expected_seed = {"gene2"};
  EXPECT_EQ(seeds[1], expected_seed);

  expected_seed = {"gene3"};
  EXPECT_EQ(seeds[2], expected_seed);

  expected_seed = {"gene4"};
  EXPECT_EQ(seeds[3], expected_seed);

  expected_seed = {"gene5"};
  EXPECT_EQ(seeds[4], expected_seed);
}

TEST(TestRwrUtils, ReadSeedsMultipleSeedsPerLineDoNotSkipFirstColumn) {
  const std::string filename = "seedfile.txt";
  write_test_file(filename,
    "gene1\tgene6\n"
    "gene2\n"
    "gene3\tgene7\tgene8\n"
    "gene4\tgene9\n"
    "gene5\tgene10\tgene11\tgene12\n"
  );

  std::vector<std::vector<std::string>> seeds;

  rwr::read_seeds(seeds, filename, false);

  std::remove(filename.c_str());

  ASSERT_EQ(seeds.size(), 5);
  std::vector<std::string> expected_seed = {"gene1", "gene6"};
  EXPECT_EQ(seeds[0], expected_seed);

  expected_seed = {"gene2"};
  EXPECT_EQ(seeds[1], expected_seed);

  expected_seed = {"gene3","gene7", "gene8"};
  EXPECT_EQ(seeds[2], expected_seed);

  expected_seed = {"gene4", "gene9"};
  EXPECT_EQ(seeds[3], expected_seed);

  expected_seed = {"gene5","gene10","gene11","gene12"};
  EXPECT_EQ(seeds[4], expected_seed);
}

TEST(TestRwrUtils, ReadSeedsMultipleSeedsPerLineDoNotSkipFirstColumnCsv) {
  const std::string filename = "seedfile.txt";
  write_test_file(filename,
    "gene1,gene6\n"
    "gene2\n"
    "gene3,gene7,gene8\n"
    "gene4,gene9\n"
    "gene5,gene10,gene11,gene12\n"
  );

  std::vector<std::vector<std::string>> seeds;

  rwr::read_seeds(seeds, filename, false, ',');

  std::remove(filename.c_str());

  ASSERT_EQ(seeds.size(), 5);
  std::vector<std::string> expected_seed = {"gene1", "gene6"};
  EXPECT_EQ(seeds[0], expected_seed);

  expected_seed = {"gene2"};
  EXPECT_EQ(seeds[1], expected_seed);

  expected_seed = {"gene3","gene7", "gene8"};
  EXPECT_EQ(seeds[2], expected_seed);

  expected_seed = {"gene4", "gene9"};
  EXPECT_EQ(seeds[3], expected_seed);

  expected_seed = {"gene5","gene10","gene11","gene12"};
  EXPECT_EQ(seeds[4], expected_seed);
}

TEST(TestRwrUtils, ReadSeedsMultipleSeedsPerLineSkipFirstColumn) {
  const std::string filename = "seedfile.txt";
  write_test_file(filename,
    "set1\tgene1\tgene6\n"
    "set1\tgene2\n"
    "set1\tgene3\tgene7\tgene8\n"
    "set2\tgene4\tgene9\n"
    "set2\tgene5\tgene10\tgene11\tgene12\n"
  );

  std::vector<std::vector<std::string>> seeds;

  rwr::read_seeds(seeds, filename);

  std::remove(filename.c_str());

  ASSERT_EQ(seeds.size(), 5);
  std::vector<std::string> expected_seed = {"gene1", "gene6"};
  EXPECT_EQ(seeds[0], expected_seed);

  expected_seed = {"gene2"};
  EXPECT_EQ(seeds[1], expected_seed);

  expected_seed = {"gene3","gene7", "gene8"};
  EXPECT_EQ(seeds[2], expected_seed);

  expected_seed = {"gene4", "gene9"};
  EXPECT_EQ(seeds[3], expected_seed);

  expected_seed = {"gene5","gene10","gene11","gene12"};
  EXPECT_EQ(seeds[4], expected_seed);
}

TEST(TestRwrUtils, WriteSeedsThrowsOnBadFileName) {
  const std::string filename = "/bad_dir/bad_seed_file.txt";
  std::vector<std::vector<std::string>> seeds;

  ASSERT_THAT(
    [&](){rwr::write_seeds(seeds, filename); },
    testing::ThrowsMessage<std::runtime_error>("rwr::write_seeds - could not open file: /bad_dir/bad_seed_file.txt")
  );
}

TEST(TestRwrUtils, WriteSeedsThrowsOnBadFirstColumnSize) {
  const std::string filename = "output_seed_file.txt";
  std::vector<std::vector<std::string>> seeds;
  seeds.push_back({"gene1"});
  seeds.push_back({"gene2"});
  seeds.push_back({"gene3"});

  std::vector<std::string> first_col = {"set1", "set1"};

  ASSERT_THAT(
    [&](){rwr::write_seeds(seeds, filename, first_col); },
    testing::ThrowsMessage<std::runtime_error>("rwr::write_seeds - first column is not empty and size does not match seed size()")
  );
}

TEST(TestRwrUtils, WriteSeedsSingleSeedPerLineWithFirstColumn) {
  const std::string filename = "seedfile.txt";
  std::vector<std::vector<std::string>> seeds;
  seeds.push_back({"gene1"});
  seeds.push_back({"gene2"});
  seeds.push_back({"gene3"});
  seeds.push_back({"gene4"});

  std::vector<std::string> first_col = {"set1", "set1", "set2", "set2"};

  rwr::write_seeds(seeds, filename, first_col);

  std::ifstream file(filename);
  ASSERT_TRUE(file.is_open()) << "Could not open output file";

  std::string line;
  std::vector<std::string> lines;
  while (std::getline(file, line)) {
    lines.push_back(line);
  }
  file.close();
  std::remove(filename.c_str());

  ASSERT_EQ(lines.size(), 4);
  EXPECT_EQ(lines[0], "set1\tgene1");
  EXPECT_EQ(lines[1], "set1\tgene2");
  EXPECT_EQ(lines[2], "set2\tgene3");
  EXPECT_EQ(lines[3], "set2\tgene4");
}

TEST(TestRwrUtils, WriteSeedsSingleSeedPerLineWithoutFirstColumn) {
    const std::string filename = "seedfile.txt";
  std::vector<std::vector<std::string>> seeds;
  seeds.push_back({"gene1"});
  seeds.push_back({"gene2"});
  seeds.push_back({"gene3"});
  seeds.push_back({"gene4"});

  rwr::write_seeds(seeds, filename);

  std::ifstream file(filename);
  ASSERT_TRUE(file.is_open()) << "Could not open output file";

  std::string line;
  std::vector<std::string> lines;
  while (std::getline(file, line)) {
    lines.push_back(line);
  }
  file.close();
  std::remove(filename.c_str());

  ASSERT_EQ(lines.size(), 4);
  EXPECT_EQ(lines[0], "gene1");
  EXPECT_EQ(lines[1], "gene2");
  EXPECT_EQ(lines[2], "gene3");
  EXPECT_EQ(lines[3], "gene4");
}

TEST(TestRwrUtils, WriteSeedsMultipleSeedsPerLineWithFirstColumn) {
  const std::string filename = "seedfile.txt";
  std::vector<std::vector<std::string>> seeds;
  seeds.push_back({"gene1", "gene5"});
  seeds.push_back({"gene2"});
  seeds.push_back({"gene3", "gene6", "gene7"});
  seeds.push_back({"gene4", "gene8", "gene9", "gene10"});

  std::vector<std::string> first_col = {"set1", "set1", "set2", "set2"};

  rwr::write_seeds(seeds, filename, first_col);

  std::ifstream file(filename);
  ASSERT_TRUE(file.is_open()) << "Could not open output file";

  std::string line;
  std::vector<std::string> lines;
  while (std::getline(file, line)) {
    lines.push_back(line);
  }
  file.close();
  std::remove(filename.c_str());

  ASSERT_EQ(lines.size(), 4);
  EXPECT_EQ(lines[0], "set1\tgene1\tgene5");
  EXPECT_EQ(lines[1], "set1\tgene2");
  EXPECT_EQ(lines[2], "set2\tgene3\tgene6\tgene7");
  EXPECT_EQ(lines[3], "set2\tgene4\tgene8\tgene9\tgene10");
}

TEST(TestRwrUtils, WriteSeedsMultipleSeedsPerLineWithoutFirstColumn) {
  const std::string filename = "seedfile.txt";
  std::vector<std::vector<std::string>> seeds;
  seeds.push_back({"gene1", "gene5"});
  seeds.push_back({"gene2"});
  seeds.push_back({"gene3", "gene6", "gene7"});
  seeds.push_back({"gene4", "gene8", "gene9", "gene10"});

  rwr::write_seeds(seeds, filename);

  std::ifstream file(filename);
  ASSERT_TRUE(file.is_open()) << "Could not open output file";

  std::string line;
  std::vector<std::string> lines;
  while (std::getline(file, line)) {
    lines.push_back(line);
  }
  file.close();
  std::remove(filename.c_str());

  ASSERT_EQ(lines.size(), 4);
  EXPECT_EQ(lines[0], "gene1\tgene5");
  EXPECT_EQ(lines[1], "gene2");
  EXPECT_EQ(lines[2], "gene3\tgene6\tgene7");
  EXPECT_EQ(lines[3], "gene4\tgene8\tgene9\tgene10");
}

TEST(TestRwrUtils, GetDefaultSeedSetEmptyNodeLabel) {
  std::vector<std::vector<std::string>> seed_set;
  std::vector<std::string> node_labels;

  rwr::get_default_seed_set(seed_set, node_labels);

  EXPECT_EQ(seed_set.size(), 0);
}

TEST(TestRwrUtils, GetDefaultSeedSetNonEmptyNodeLabel) {
  std::vector<std::vector<std::string>> seed_set;
  std::vector<std::string> node_labels = {"node1", "node2", "node3", "node4"};

  rwr::get_default_seed_set(seed_set, node_labels);

  ASSERT_EQ(seed_set.size(), 4);
  ASSERT_EQ(seed_set[0].size(), 1);
  EXPECT_EQ(seed_set[0][0], "node1");
  ASSERT_EQ(seed_set[1].size(), 1);
  EXPECT_EQ(seed_set[1][0], "node2");
  ASSERT_EQ(seed_set[2].size(), 1);
  EXPECT_EQ(seed_set[2][0], "node3");
  ASSERT_EQ(seed_set[3].size(), 1);
  EXPECT_EQ(seed_set[3][0], "node4");
}

TEST(TestRwrUtils, GetGrinLooSeeds) {
  std::vector<std::vector<std::string>> seed_set;
  std::vector<std::string> loo_seeds;

  std::vector<std::string> base_seeds = {"node1", "node2", "node3", "node4"};
  std::vector<std::string> node_labels = {"node1", "node2", "node3", "node4",
                                          "node5", "node6", "node7", "node8",
                                          "node9", "node10"};
  const std::size_t n_samples_in_null_dist = 10;

  rwr::get_grin_loo_seeds(seed_set, loo_seeds, base_seeds, node_labels, n_samples_in_null_dist);

  std::unordered_set<std::string> allowed_set = {"node1", "node2", "node3", "node4",
                                                 "node5", "node6", "node7", "node8",
                                                  "node9", "node10"};

  ASSERT_EQ(seed_set.size(), 44);
  for (std::size_t i = 0; i < 44; ++i) {
    ASSERT_EQ(seed_set[i].size(), 3);
  }

  EXPECT_EQ(seed_set[0][0], "node2");
  EXPECT_EQ(seed_set[0][1], "node3");
  EXPECT_EQ(seed_set[0][2], "node4");

  EXPECT_EQ(seed_set[1][0], "node1");
  EXPECT_EQ(seed_set[1][1], "node3");
  EXPECT_EQ(seed_set[1][2], "node4");

  EXPECT_EQ(seed_set[2][0], "node1");
  EXPECT_EQ(seed_set[2][1], "node2");
  EXPECT_EQ(seed_set[2][2], "node4");

  EXPECT_EQ(seed_set[3][0], "node1");
  EXPECT_EQ(seed_set[3][1], "node2");
  EXPECT_EQ(seed_set[3][2], "node3");

  for (std::size_t i = 4; i < 44; ++i) {
    std::unordered_set<std::string> seen;

    for (const auto& seed : seed_set[i]) {
      EXPECT_TRUE(allowed_set.count(seed)); // found in node labels

      bool inserted = seen.insert(seed).second;
      EXPECT_TRUE(inserted); // Unique in current vector
    }
  }
  
  ASSERT_EQ(loo_seeds.size(), 44);

  EXPECT_EQ(loo_seeds[0], "node1");
  EXPECT_EQ(loo_seeds[1], "node2");
  EXPECT_EQ(loo_seeds[2], "node3");
  EXPECT_EQ(loo_seeds[3], "node4");

  for (std::size_t i = 4; i < 44; ++i) {
    EXPECT_TRUE(allowed_set.count(loo_seeds[i]));
  }
}

TEST(TestRwrUtils, GetEncodingSizeThrowsOnUnknownMethod) {
  ASSERT_THAT(
    [&](){rwr::get_encoding_size("bad_method", 100, 5); },
    testing::ThrowsMessage<std::invalid_argument>("rwr::get_encoding_size - unknown reduction method")
  );
}

TEST(TestRwrUtils, GetEncodingSizeForGeometicMethod) {
  const std::size_t N = 500;
  const std::size_t L = 10;
  std::size_t actual_size = rwr::get_encoding_size("geometric", N, L);
  
  EXPECT_EQ(actual_size, N);
}

TEST(TestRwrUtils, GetEncodingSizeForArithmeticMethod) {
  const std::size_t N = 500;
  const std::size_t L = 10;
  std::size_t actual_size = rwr::get_encoding_size("arithmetic", N, L);
  
  EXPECT_EQ(actual_size, N);
}

TEST(TestRwrUtils, GetEncodingSizeForSumMethod) {
  const std::size_t N = 500;
  const std::size_t L = 10;
  std::size_t actual_size = rwr::get_encoding_size("sum", N, L);
  
  EXPECT_EQ(actual_size, N);
}

TEST(TestRwrUtils, GetEncodingSizeForNoneMethod) {
  const std::size_t N = 500;
  const std::size_t L = 10;
  std::size_t actual_size = rwr::get_encoding_size("none", N, L);
  
  EXPECT_EQ(actual_size, N * L);
}

TEST(TestRwrUtils, GetInitProbsThrowsOnTauSizeMismatch) {
  std::vector<std::vector<std::string>> seed_set;
  seed_set.push_back({"node1"});
  seed_set.push_back({"node2", "node3"});
  seed_set.push_back({"node10"});

  const std::vector<std::string> nodes_in_mp = {"node1", "node2", "node3",
                                                "node4", "node5", "node6",
                                                "node7", "node8", "node9"};

  const std::size_t L = 3;
  const std::vector<double> tau = {0.5, 0.5};

  ASSERT_THAT(
    [&](){rwr::get_init_probs(seed_set, nodes_in_mp, L, tau); },
    testing::ThrowsMessage<std::invalid_argument>("rwr::get_init_probs - tau vector must be empty or have length of L")
  );
}

TEST(TestRwrUtils, GetInitProbsThrowsOnBadTauSum) {
  std::vector<std::vector<std::string>> seed_set;
  seed_set.push_back({"node1"});
  seed_set.push_back({"node2", "node3"});
  seed_set.push_back({"node10"});

  const std::vector<std::string> nodes_in_mp = {"node1", "node2", "node3",
                                                "node4", "node5", "node6",
                                                "node7", "node8", "node9"};

  const std::size_t L = 3;
  const std::vector<double> tau = {0.5, 0.5, 0.5};

  ASSERT_THAT(
    [&](){rwr::get_init_probs(seed_set, nodes_in_mp, L, tau); },
    testing::ThrowsMessage<std::invalid_argument>("rwr::get_init_probs - elements of tau must sum to 1.0")
  );
}

TEST(TestRwrUtils, GetInitProbsReturnsEmptyOnN_EqualToZero) {
  std::vector<std::vector<std::string>> seed_set;
  seed_set.push_back({"node1"});
  seed_set.push_back({"node2", "node3"});
  seed_set.push_back({"node10"});

  const std::vector<std::string> nodes_in_mp ;

  const std::size_t L = 3;
  const std::vector<double> tau;

  auto actual_probs = rwr::get_init_probs(seed_set, nodes_in_mp, L, tau);

  EXPECT_TRUE(actual_probs.empty());
}

TEST(TestRwrUtils, GetInitProbsReturnsEmptyOnL_EqualToZero) {
  std::vector<std::vector<std::string>> seed_set;
  seed_set.push_back({"node1"});
  seed_set.push_back({"node2", "node3"});
  seed_set.push_back({"node10"});

  const std::vector<std::string> nodes_in_mp ;

  const std::size_t L = 0;
  const std::vector<double> tau;

  auto actual_probs = rwr::get_init_probs(seed_set, nodes_in_mp, L, tau);

  EXPECT_TRUE(actual_probs.empty());
}

TEST(TestRwrUtils, GetInitProbsReturnsEmptyOnNumSeedsEqualToZero) {
  std::vector<std::vector<std::string>> seed_set;
  const std::vector<std::string> nodes_in_mp ;

  const std::size_t L = 3;
  const std::vector<double> tau;

  auto actual_probs = rwr::get_init_probs(seed_set, nodes_in_mp, L, tau);

  EXPECT_TRUE(actual_probs.empty());
}

TEST(TestRwrUtils, GetInitProbsReturnsCorrectProbDefaultTau) {
  std::vector<std::vector<std::string>> seed_set;
  seed_set.push_back({"node1"});
  seed_set.push_back({"node2", "node3"});
  seed_set.push_back({"node10"});

  const std::vector<std::string> nodes_in_mp = {"node1", "node2", "node3",
                                                "node4", "node5", "node6",
                                                "node7", "node8", "node9"};

  const std::size_t L = 3;
  
  auto actual_init_probs = rwr::get_init_probs(seed_set, nodes_in_mp, L);

  ASSERT_EQ(actual_init_probs.size(), 81);

  for (std::size_t i = 0; i < 81; ++i) {
    if (i == 0 || i == 9 || i == 18) {
      EXPECT_NEAR(actual_init_probs[i], 1.0/3, 1e-8);
    } else if (i == 28 || i == 29 || i == 37 || i == 38 || i == 46 || i == 47) {
      EXPECT_NEAR(actual_init_probs[i], 1.0/6, 1e-8);
    } else {
      EXPECT_EQ(actual_init_probs[i], 0.0);
    }
  }
}

TEST(TestRwrUtils, GetInitProbsReturnsCorrectProbNonDefaultTau) {
  std::vector<std::vector<std::string>> seed_set;
  seed_set.push_back({"node1"});
  seed_set.push_back({"node2", "node3"});
  seed_set.push_back({"node10"});

  const std::vector<std::string> nodes_in_mp = {"node1", "node2", "node3",
                                                "node4", "node5", "node6",
                                                "node7", "node8", "node9"};

  const std::size_t L = 3;
  const std::vector<double> tau = {0.5, 0.25, 0.25};
  auto actual_init_probs = rwr::get_init_probs(seed_set, nodes_in_mp, L, tau);

  ASSERT_EQ(actual_init_probs.size(), 81);

  for (std::size_t i = 0; i < 81; ++i) {
    if (i == 0) {
      EXPECT_NEAR(actual_init_probs[i], 1.0/2, 1e-8);
    } else if ( i == 9 || i == 18) {
      EXPECT_NEAR(actual_init_probs[i], 1.0/4, 1e-8);
    } else if (i == 28 || i == 29) {
      EXPECT_NEAR(actual_init_probs[i], 1.0/4, 1e-8);
    } else if (i == 37 || i == 38 || i == 46 || i == 47) {
      EXPECT_NEAR(actual_init_probs[i], 1.0/8, 1e-8);
    } else {
      EXPECT_EQ(actual_init_probs[i], 0.0);
    }
  }
}

TEST(TestRwrUtils, FlattenSingleElement) {
  std::vector<std::vector<std::string>> input;
  input.push_back({"node1"});
  input.push_back({"node2"});

  auto output = rwr::flatten(input);

  ASSERT_EQ(output.size(), 2);
  EXPECT_EQ(output[0], "node1");
  EXPECT_EQ(output[1], "node2");
}

TEST(TestRwrUtils, FlattenMultipleElement) {
  std::vector<std::vector<std::string>> input;
  input.push_back({"node1", "node2"});
  input.push_back({"node3", "node4", "node5"});

  auto output = rwr::flatten(input);

  ASSERT_EQ(output.size(), 2);
  EXPECT_EQ(output[0], "node1_node2");
  EXPECT_EQ(output[1], "node3_node4_node5");
}

TEST(TestRwrUtils, RemoveSeedsNotInListEmptyList) {
  std::vector<std::vector<std::string>> seed_set;
  seed_set.push_back({"node1", "node2"});
  seed_set.push_back({"node3"});

  rwr::remove_seeds_not_in_list(seed_set, {});

  ASSERT_EQ(seed_set.size(), 2);
  ASSERT_EQ(seed_set[0].size(), 2);
  ASSERT_EQ(seed_set[1].size(), 1);
  EXPECT_EQ(seed_set[0][0], "node1");
  EXPECT_EQ(seed_set[0][1], "node2");
  EXPECT_EQ(seed_set[1][0], "node3");
}

TEST(TestRwrUtils, RemoveSeedsNotInListRemoveNoSeeds) {
  std::vector<std::vector<std::string>> seed_set;
  seed_set.push_back({"node1", "node2"});
  seed_set.push_back({"node3"});

  std::vector<std::string> x = {"node1", "node2", "node3"};
  rwr::remove_seeds_not_in_list(seed_set, x);

  ASSERT_EQ(seed_set.size(), 2);
  ASSERT_EQ(seed_set[0].size(), 2);
  ASSERT_EQ(seed_set[1].size(), 1);
  EXPECT_EQ(seed_set[0][0], "node1");
  EXPECT_EQ(seed_set[0][1], "node2");
  EXPECT_EQ(seed_set[1][0], "node3");
}

TEST(TestRwrUtils, RemoveSeedsNotInListRemoveSeed) {
  std::vector<std::vector<std::string>> seed_set;
  seed_set.push_back({"node1", "node2"});
  seed_set.push_back({"node3"});

  std::vector<std::string> x = {"node1", "node3"};
  rwr::remove_seeds_not_in_list(seed_set, x);

  ASSERT_EQ(seed_set.size(), 2);
  ASSERT_EQ(seed_set[0].size(), 1);
  ASSERT_EQ(seed_set[1].size(), 1);
  EXPECT_EQ(seed_set[0][0], "node1");
  EXPECT_EQ(seed_set[1][0], "node3");
}

TEST(TestRwrUtils, RemoveSeedsNotInListRemoveSeedVector) {
  std::vector<std::vector<std::string>> seed_set;
  seed_set.push_back({"node1", "node2"});
  seed_set.push_back({"node3"});
  seed_set.push_back({"node4"});

  std::vector<std::string> x = {"node1", "node2", "node4"};
  rwr::remove_seeds_not_in_list(seed_set, x);

  ASSERT_EQ(seed_set.size(), 2);
  ASSERT_EQ(seed_set[0].size(), 2);
  EXPECT_EQ(seed_set[0][0], "node1");
  EXPECT_EQ(seed_set[0][1], "node2");
  ASSERT_EQ(seed_set[1].size(), 1);
  EXPECT_EQ(seed_set[1][0], "node4");
}

TEST(TestRwrUtils, RemoveDuplicateSeedVectorsNoDups) {
  std::vector<std::vector<std::string>> seed_set;
  seed_set.push_back({"node1", "node2"});
  seed_set.push_back({"node3"});
  seed_set.push_back({"node4"});

  rwr::remove_duplicate_seed_vectors(seed_set);

  ASSERT_EQ(seed_set.size(), 3);
  ASSERT_EQ(seed_set[0].size(), 2);
  EXPECT_EQ(seed_set[0][0], "node1");
  EXPECT_EQ(seed_set[0][1], "node2");
  ASSERT_EQ(seed_set[1].size(), 1);
  EXPECT_EQ(seed_set[1][0], "node3");
  ASSERT_EQ(seed_set[2].size(), 1);
  EXPECT_EQ(seed_set[2][0], "node4");
}

TEST(TestRwrUtils, RemoveDuplicateSeedVectorsDups) {
  std::vector<std::vector<std::string>> seed_set;
  seed_set.push_back({"node1", "node2"});
  seed_set.push_back({"node3", "node4"});
  seed_set.push_back({"node3", "node4"});
  seed_set.push_back({"node5"});
  seed_set.push_back({"node5", "node6"});

  rwr::remove_duplicate_seed_vectors(seed_set);

  ASSERT_EQ(seed_set.size(), 4);
  ASSERT_EQ(seed_set[0].size(), 2);
  EXPECT_EQ(seed_set[0][0], "node1");
  EXPECT_EQ(seed_set[0][1], "node2");
  ASSERT_EQ(seed_set[1].size(), 2);
  EXPECT_EQ(seed_set[1][0], "node3");
  EXPECT_EQ(seed_set[1][1], "node4");
  ASSERT_EQ(seed_set[2].size(), 1);
  EXPECT_EQ(seed_set[2][0], "node5");
  ASSERT_EQ(seed_set[3].size(), 2);
  EXPECT_EQ(seed_set[3][0], "node5");
  EXPECT_EQ(seed_set[3][1], "node6");
}

TEST(TestRwrUtils, GetSeedSplitForAllWorkersAllValid) {
  std::vector<unsigned long> start_seed_vec, stop_seed_vec, num_encodings;

  rwr::get_seed_split_for_all_workers(8, 10, start_seed_vec, stop_seed_vec, num_encodings);

  ASSERT_EQ(start_seed_vec.size(), 8);
  ASSERT_EQ(stop_seed_vec.size(), 8);
  ASSERT_EQ(num_encodings.size(), 8);

  EXPECT_EQ(start_seed_vec[0], 0);
  EXPECT_EQ(start_seed_vec[1], 2);
  EXPECT_EQ(start_seed_vec[2], 4);
  EXPECT_EQ(start_seed_vec[3], 5);
  EXPECT_EQ(start_seed_vec[4], 6);
  EXPECT_EQ(start_seed_vec[5], 7);
  EXPECT_EQ(start_seed_vec[6], 8);
  EXPECT_EQ(start_seed_vec[7], 9);

  EXPECT_EQ(stop_seed_vec[0], 1);
  EXPECT_EQ(stop_seed_vec[1], 3);
  EXPECT_EQ(stop_seed_vec[2], 4);
  EXPECT_EQ(stop_seed_vec[3], 5);
  EXPECT_EQ(stop_seed_vec[4], 6);
  EXPECT_EQ(stop_seed_vec[5], 7);
  EXPECT_EQ(stop_seed_vec[6], 8);
  EXPECT_EQ(stop_seed_vec[7], 9);

  EXPECT_EQ(num_encodings[0], 2);
  EXPECT_EQ(num_encodings[1], 2);
  EXPECT_EQ(num_encodings[2], 1);
  EXPECT_EQ(num_encodings[3], 1);
  EXPECT_EQ(num_encodings[4], 1);
  EXPECT_EQ(num_encodings[5], 1);
  EXPECT_EQ(num_encodings[6], 1);
  EXPECT_EQ(num_encodings[7], 1);
}

TEST(TestRwrUtils, GetSeedSplitForAllWorkersNotAllValid) {
  std::vector<unsigned long> start_seed_vec, stop_seed_vec, num_encodings;

  rwr::get_seed_split_for_all_workers(8, 6, start_seed_vec, stop_seed_vec, num_encodings);

  ASSERT_EQ(start_seed_vec.size(), 8);
  ASSERT_EQ(stop_seed_vec.size(), 8);
  ASSERT_EQ(num_encodings.size(), 8);

  EXPECT_EQ(start_seed_vec[0], 0);
  EXPECT_EQ(start_seed_vec[1], 1);
  EXPECT_EQ(start_seed_vec[2], 2);
  EXPECT_EQ(start_seed_vec[3], 3);
  EXPECT_EQ(start_seed_vec[4], 4);
  EXPECT_EQ(start_seed_vec[5], 5);
  EXPECT_EQ(start_seed_vec[6], 0);
  EXPECT_EQ(start_seed_vec[7], 0);

  EXPECT_EQ(stop_seed_vec[0], 0);
  EXPECT_EQ(stop_seed_vec[1], 1);
  EXPECT_EQ(stop_seed_vec[2], 2);
  EXPECT_EQ(stop_seed_vec[3], 3);
  EXPECT_EQ(stop_seed_vec[4], 4);
  EXPECT_EQ(stop_seed_vec[5], 5);
  EXPECT_EQ(stop_seed_vec[6], 0);
  EXPECT_EQ(stop_seed_vec[7], 0);

  EXPECT_EQ(num_encodings[0], 1);
  EXPECT_EQ(num_encodings[1], 1);
  EXPECT_EQ(num_encodings[2], 1);
  EXPECT_EQ(num_encodings[3], 1);
  EXPECT_EQ(num_encodings[4], 1);
  EXPECT_EQ(num_encodings[5], 1);
  EXPECT_EQ(num_encodings[6], 0);
  EXPECT_EQ(num_encodings[7], 0);
}

TEST(TestRwrUtils, ScaleNodeValuesByZero) {
  std::vector<double> values = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  std::vector<int32_t> col_idx = {0, 1, 2, 0, 1, 2};
  std::vector<int32_t> row_ptr = {0, 3, 6};
  CSR_Matrix mat(2, 3, values, col_idx, row_ptr);

  rwr::scale_node_values(mat, 3, 1, 0.0);

  EXPECT_EQ(mat.col_idx_, col_idx);
  EXPECT_EQ(mat.row_ptr_, row_ptr);
  std::vector<double> expected_values = {1.0, 0.0, 3.0, 0.0, 0.0, 0.0};
  EXPECT_EQ(mat.values_, expected_values);
}

TEST(TestRwrUtils, ScaleNodeValuesByTwo) {
  std::vector<double> values = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  std::vector<int32_t> col_idx = {0, 1, 2, 0, 1, 2};
  std::vector<int32_t> row_ptr = {0, 3, 6};
  CSR_Matrix mat(2, 3, values, col_idx, row_ptr);

  rwr::scale_node_values(mat, 3, 0, 2.0);

  EXPECT_EQ(mat.col_idx_, col_idx);
  EXPECT_EQ(mat.row_ptr_, row_ptr);
  std::vector<double> expected_values = {2.0, 4.0, 6.0, 8.0, 5.0, 6.0};
  EXPECT_EQ(mat.values_, expected_values);
}

}
